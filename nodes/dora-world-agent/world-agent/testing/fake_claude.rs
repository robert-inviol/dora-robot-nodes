//! A stand-in for `claude -p` in stream-json mode, for the tests. Each process plays the model `--model` names, and
//! lines of its user message of the form `fake <model> <step>` script that model. The steps run in order:
//!
//! - `wait <ms>` streams text deltas for that long, as a model writing, still honouring an interrupt;
//! - `call <tool> <json arguments>` calls a world tool on the MCP endpoint of `--mcp-config`, with the agent id and
//!   token from the environment; a tool error ends the call as an error;
//! - `say <text>` ends the call with that answer;
//! - `hang` never answers but honours an interrupt; `deaf` never answers and ignores interrupts;
//! - `crash` exits at once with status 3; `auth` ends the call with API error status 401;
//! - `argv` answers with its arguments and the environment variables the client sets, as JSON.
//!
//! `fake <model> after <other model> <step>` applies only to a reconciler prompt whose later answer is from the other
//! model. A model with no script answers `echo: <message>`, or the text of `[fake:reply=<text>]` in its system prompt.
//! `[fake:permission=<mode>]` in the system prompt sets the permission mode its init event reports.

use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use world_agent::config::ModelName;
use world_agent::reconcile::later_answer_marker;

const DELTA_EVERY: Duration = Duration::from_millis(50);
const REPORTED_ENVIRONMENT: [&str; 6] = [
    "AGENT_ID",
    "ENABLE_TOOL_SEARCH",
    "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
    "DISABLE_AUTOUPDATER",
    "CLAUDE_CODE_RESUME_INTERRUPTED_TURN",
];
const TOOLS: [&str; 10] = [
    "find",
    "describe_object",
    "clearance_between",
    "route",
    "last_seen",
    "changes_since",
    "render_view",
    "query_sql",
    "send_goal",
    "cancel_goal",
];

fn emit(value: Value) {
    let mut out = std::io::stdout().lock();
    writeln!(out, "{value}").expect("writing stdout");
    out.flush().expect("flushing stdout");
}

fn argument_after(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned()
}

fn directive(prompt: &str, name: &str) -> Option<String> {
    let start = prompt.find(&format!("[fake:{name}="))? + name.len() + 7;
    let end = start + prompt[start..].find(']')?;
    Some(prompt[start..end].to_owned())
}

/// The world server of `--mcp-config`, spoken to over plain HTTP/1.1.
struct McpServer {
    authority: String,
    path: String,
    headers: Vec<(String, String)>,
}

/// One HTTP response: its status, its headers with lower-case names, and its body.
struct HttpReply {
    status: u16,
    headers: Vec<(String, String)>,
    body: String,
}

fn expand_environment(value: &str) -> String {
    let mut expanded = value.to_owned();
    for name in ["AGENT_ID", "AGENT_TOKEN"] {
        expanded = expanded.replace(&format!("${{{name}}}"), &std::env::var(name).unwrap_or_default());
    }
    expanded
}

/// The JSON-RPC message with this id in a JSON or server-sent-events body.
fn rpc_message(body: &str, id: u64) -> Option<Value> {
    let candidates = std::iter::once(body).chain(body.lines().filter_map(|l| l.strip_prefix("data:")));
    candidates.filter_map(|text| serde_json::from_str::<Value>(text.trim()).ok()).find(|m| m["id"] == id)
}

fn read_body(reader: &mut BufReader<TcpStream>, headers: &[(String, String)], expected_id: Option<u64>) -> String {
    let header = |name: &str| headers.iter().find(|(n, _)| n == name).map(|(_, v)| v.as_str());
    if let Some(length) = header("content-length") {
        let mut body = vec![0; length.parse().expect("a content length")];
        reader.read_exact(&mut body).expect("reading the body");
        return String::from_utf8(body).expect("a UTF-8 body");
    }
    let mut body = String::new();
    if header("transfer-encoding") == Some("chunked") {
        loop {
            let mut size = String::new();
            reader.read_line(&mut size).expect("reading a chunk size");
            let size = usize::from_str_radix(size.trim(), 16).expect("a hex chunk size");
            if size == 0 {
                return body;
            }
            let mut chunk = vec![0; size + 2];
            reader.read_exact(&mut chunk).expect("reading a chunk");
            body.push_str(std::str::from_utf8(&chunk[..size]).expect("a UTF-8 chunk"));
            if expected_id.is_some_and(|id| rpc_message(&body, id).is_some()) {
                return body;
            }
        }
    }
    reader.read_to_string(&mut body).expect("reading the body");
    body
}

impl McpServer {
    fn from_arguments(args: &[String]) -> Option<Self> {
        let config: Value = serde_json::from_str(&argument_after(args, "--mcp-config")?).expect("JSON MCP config");
        let world = &config["mcpServers"]["world"];
        let url = world["url"].as_str().expect("a world URL");
        let rest = url.strip_prefix("http://").expect("an http URL");
        let (authority, path) = rest.split_at(rest.find('/').expect("a path"));
        let headers = world["headers"]
            .as_object()
            .expect("headers")
            .iter()
            .map(|(name, value)| (name.clone(), expand_environment(value.as_str().expect("a header value"))))
            .collect();
        Some(Self { authority: authority.to_owned(), path: path.to_owned(), headers })
    }

    fn post(&self, session: Option<&str>, message: &Value) -> HttpReply {
        let mut stream = TcpStream::connect(&self.authority).expect("connecting to the MCP endpoint");
        let body = message.to_string();
        let mut request = format!(
            "POST {} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nAccept: application/json, text/event-stream\r\n\
             Content-Length: {}\r\nConnection: close\r\n",
            self.path,
            self.authority,
            body.len()
        );
        for (name, value) in &self.headers {
            request.push_str(&format!("{name}: {value}\r\n"));
        }
        if let Some(session) = session {
            request.push_str(&format!("Mcp-Session-Id: {session}\r\n"));
        }
        request.push_str("\r\n");
        request.push_str(&body);
        stream.write_all(request.as_bytes()).expect("writing the request");
        let mut reader = BufReader::new(stream);
        let mut status_line = String::new();
        reader.read_line(&mut status_line).expect("reading the status line");
        let status = status_line.split_whitespace().nth(1).and_then(|s| s.parse().ok()).expect("an HTTP status");
        let mut headers = Vec::new();
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).expect("reading a header");
            let line = line.trim_end();
            if line.is_empty() {
                break;
            }
            let (name, value) = line.split_once(':').expect("a header line");
            headers.push((name.trim().to_ascii_lowercase(), value.trim().to_owned()));
        }
        let expected_id = message["id"].as_u64();
        let body = read_body(&mut reader, &headers, expected_id);
        HttpReply { status, headers, body }
    }

    /// Calls a tool and returns its text, or why it failed.
    fn call_tool(&self, tool: &str, arguments: Value) -> Result<String, String> {
        let initialize = json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "fake-claude", "version": "1"}}});
        let reply = self.post(None, &initialize);
        if reply.status != 200 {
            return Err(format!("initialize: HTTP {} {}", reply.status, reply.body));
        }
        let session = reply.headers.iter().find(|(n, _)| n == "mcp-session-id").map(|(_, v)| v.clone());
        self.post(session.as_deref(), &json!({"jsonrpc": "2.0", "method": "notifications/initialized"}));
        let call = json!({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": tool, "arguments": arguments}});
        let reply = self.post(session.as_deref(), &call);
        let message = rpc_message(&reply.body, 2).ok_or_else(|| format!("{tool}: HTTP {} {}", reply.status, reply.body))?;
        if let Some(error) = message.get("error") {
            return Err(format!("{tool}: {error}"));
        }
        let result = &message["result"];
        let text: Vec<&str> = result["content"].as_array().into_iter().flatten().filter_map(|c| c["text"].as_str()).collect();
        match result["isError"].as_bool() {
            Some(true) => Err(format!("{tool}: {}", text.join(" "))),
            _ => Ok(text.join(" ")),
        }
    }
}

#[derive(Debug)]
enum Step {
    Wait(Duration),
    Call { tool: String, arguments: Value },
    Say(String),
    Hang,
    Deaf,
    Crash,
    Auth,
    Argv,
}

fn parse_step(text: &str) -> Step {
    let (verb, rest) = text.split_once(' ').unwrap_or((text, ""));
    match verb {
        "wait" => Step::Wait(Duration::from_millis(rest.trim().parse().expect("wait <ms>"))),
        "call" => {
            let (tool, arguments) = rest.split_once(' ').unwrap_or((rest, "{}"));
            Step::Call { tool: tool.to_owned(), arguments: serde_json::from_str(arguments).expect("call <tool> <json>") }
        }
        "say" => Step::Say(rest.to_owned()),
        "hang" => Step::Hang,
        "deaf" => Step::Deaf,
        "crash" => Step::Crash,
        "auth" => Step::Auth,
        "argv" => Step::Argv,
        other => panic!("unknown fake step {other:?}"),
    }
}

/// This model's steps in a message.
fn script(message: &str, model: &str) -> VecDeque<Step> {
    let prefix = format!("fake {model} ");
    message
        .lines()
        .filter_map(|line| line.trim().strip_prefix(&prefix))
        .filter_map(|step| match step.strip_prefix("after ") {
            Some(conditional) => {
                let (other, step) = conditional.split_once(' ').expect("after <model> <step>");
                message.contains(&later_answer_marker(&ModelName(other.to_owned()))).then_some(step)
            }
            None => Some(step),
        })
        .map(parse_step)
        .collect()
}

struct Running {
    uuid: String,
    steps: VecDeque<Step>,
    resume_at: Instant,
    last_delta: Instant,
    /// Set once a `hang` or `deaf` step is reached: whether it honours an interrupt.
    hanging: Option<bool>,
}

struct Fake {
    args: Vec<String>,
    model: String,
    session: String,
    permission_mode: String,
    reply: Option<String>,
    mcp: Option<McpServer>,
    running: Option<Running>,
}

impl Fake {
    fn result(&mut self, uuid: &str, text: &str, subtype: &str, terminal_reason: &str, api_error_status: Option<u16>) {
        emit(json!({
            "type": "result",
            "subtype": subtype,
            "is_error": subtype != "success",
            "result": text,
            "terminal_reason": terminal_reason,
            "api_error_status": api_error_status,
            "session_id": self.session,
            "user_message_uuids": [uuid],
            "usage": {"input_tokens": 2, "output_tokens": 5, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 10},
            "total_cost_usd": 0.0001,
            "duration_ms": 5,
        }));
    }

    fn init(&self) {
        let (tools, servers) = match &self.mcp {
            Some(_) => (TOOLS.iter().map(|t| format!("mcp__world__{t}")).collect(), json!([{"name": "world", "status": "connected"}])),
            None => (Vec::new(), json!([])),
        };
        emit(json!({
            "type": "system",
            "subtype": "init",
            "session_id": self.session,
            "model": self.model,
            "tools": tools,
            "mcp_servers": servers,
            "permissionMode": self.permission_mode,
            "capabilities": ["interrupt_receipt_v1", "interrupt_send_now_v1"],
            "claude_code_version": "fake",
        }));
    }

    fn user_message(&mut self, message: &Value) {
        let uuid = message["uuid"].as_str().expect("the client sets a uuid").to_owned();
        let text = message["message"]["content"].as_str().expect("text content").to_owned();
        self.init();
        emit(json!({"type": "stream_event", "event": {"type": "message_start", "message": {}}}));
        let mut steps = script(&text, &self.model);
        if steps.is_empty() {
            steps.push_back(Step::Say(self.reply.clone().unwrap_or_else(|| format!("echo: {text}"))));
        }
        let now = Instant::now();
        self.running = Some(Running { uuid, steps, resume_at: now, last_delta: now, hanging: None });
        self.advance();
    }

    fn argv_report(&self) -> String {
        let environment: serde_json::Map<String, Value> = REPORTED_ENVIRONMENT
            .iter()
            .map(|name| (name.to_string(), std::env::var(name).map_or(Value::Null, Value::String)))
            .collect();
        json!({
            "args": self.args,
            "env": environment,
            "has_token": std::env::var("AGENT_TOKEN").is_ok_and(|t| t.len() == 64),
            "cwd": std::env::current_dir().expect("a working directory"),
        })
        .to_string()
    }

    /// Runs the script's steps whose time has come.
    fn advance(&mut self) {
        loop {
            let Some(running) = self.running.as_mut() else { return };
            if running.hanging.is_some() {
                return;
            }
            if Instant::now() < running.resume_at {
                if running.last_delta.elapsed() >= DELTA_EVERY {
                    running.last_delta = Instant::now();
                    emit(json!({"type": "stream_event", "event": {"type": "content_block_delta", "index": 0,
                        "delta": {"type": "text_delta", "text": "."}}}));
                }
                return;
            }
            let uuid = running.uuid.clone();
            let Some(step) = running.steps.pop_front() else {
                self.running = None;
                self.result(&uuid, "the fake script ended without say", "error_during_execution", "completed", None);
                return;
            };
            match step {
                Step::Wait(pause) => running.resume_at = Instant::now() + pause,
                Step::Hang => running.hanging = Some(true),
                Step::Deaf => running.hanging = Some(false),
                Step::Crash => std::process::exit(3),
                Step::Call { tool, arguments } => {
                    let outcome = match &self.mcp {
                        Some(mcp) => mcp.call_tool(&tool, arguments),
                        None => Err(format!("{tool}: this process has no MCP server")),
                    };
                    if let Err(error) = outcome {
                        self.running = None;
                        self.result(&uuid, &error, "error_during_execution", "completed", None);
                        return;
                    }
                }
                Step::Say(text) => {
                    self.running = None;
                    self.result(&uuid, &text, "success", "completed", None);
                    return;
                }
                Step::Auth => {
                    self.running = None;
                    self.result(&uuid, "Invalid API key", "error_during_execution", "completed", Some(401));
                    return;
                }
                Step::Argv => {
                    self.running = None;
                    let report = self.argv_report();
                    self.result(&uuid, &report, "success", "completed", None);
                    return;
                }
            }
        }
    }

    fn control_request(&mut self, message: &Value) {
        let request_id = message["request_id"].clone();
        let response = match message["request"]["subtype"].as_str() {
            Some("interrupt") => {
                if self.running.as_ref().is_some_and(|r| r.hanging == Some(false)) {
                    return;
                }
                emit(json!({"type": "control_response", "response": {"subtype": "success", "request_id": request_id, "response": {"still_queued": []}}}));
                if let Some(running) = self.running.take() {
                    self.result(&running.uuid, "", "error_during_execution", "aborted_streaming", None);
                }
                return;
            }
            Some("mcp_status") => match &self.mcp {
                Some(_) => json!({"mcpServers": [{"name": "world", "status": "connected"}]}),
                None => json!({"mcpServers": []}),
            },
            other => {
                emit(json!({"type": "control_response", "response": {"subtype": "error", "request_id": request_id, "error": format!("unsupported {other:?}")}}));
                return;
            }
        };
        emit(json!({"type": "control_response", "response": {"subtype": "success", "request_id": request_id, "response": response}}));
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let prompt = argument_after(&args, "--system-prompt").unwrap_or_default();
    let mut fake = Fake {
        model: argument_after(&args, "--model").expect("the client sets --model"),
        session: argument_after(&args, "--session-id").expect("the client sets --session-id"),
        permission_mode: directive(&prompt, "permission").unwrap_or_else(|| argument_after(&args, "--permission-mode").unwrap_or_default()),
        reply: directive(&prompt, "reply"),
        mcp: McpServer::from_arguments(&args),
        running: None,
        args,
    };
    let (lines, input) = mpsc::channel();
    std::thread::spawn(move || {
        for line in std::io::stdin().lock().lines() {
            if lines.send(line.expect("reading stdin")).is_err() {
                return;
            }
        }
    });
    loop {
        match input.recv_timeout(Duration::from_millis(10)) {
            Ok(line) => {
                let message: Value = serde_json::from_str(&line).expect("a JSON input line");
                match message["type"].as_str() {
                    Some("user") => fake.user_message(&message),
                    Some("control_request") => fake.control_request(&message),
                    other => panic!("unexpected input type {other:?}"),
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => fake.advance(),
            Err(mpsc::RecvTimeoutError::Disconnected) => return,
        }
    }
}
