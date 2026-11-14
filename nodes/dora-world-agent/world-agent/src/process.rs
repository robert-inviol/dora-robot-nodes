//! One `claude` child process: how it is started, its input, and its output as typed events.

use std::ffi::OsString;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use nix::sys::signal::{kill, Signal};
use nix::unistd::Pid;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::mpsc;

use crate::config::{ClaudeConfig, ClaudeCredentials, Effort, ModelName, ModelProfile, SystemPrompt};
use crate::identity::{AgentId, AgentRegistry};
use crate::mcp::{McpUrl, AGENT_ID_HEADER, SERVER_NAME};
use crate::protocol::{
    ControlRequestBody, ControlResponseBody, InputMessage, McpStatusReport, OutputEvent, ProtocolError, RequestId,
    SessionId,
};

/// The environment every claude process gets on top of the supervisor's own, which carries the credentials.
const LEAN_ENVIRONMENT: [(&str, &str); 9] = [
    // MCP tools load up front; otherwise the first use of each costs a ToolSearch round trip.
    ("ENABLE_TOOL_SEARCH", "false"),
    // A "now" message aborts a running tool call instead of moving it to the background.
    ("CLAUDE_CODE_DISABLE_BACKGROUND_TASKS", "1"),
    ("CLAUDE_CODE_DISABLE_AUTO_MEMORY", "1"),
    ("CLAUDE_CODE_DISABLE_CLAUDE_MDS", "1"),
    ("CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS", "1"),
    ("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1"),
    ("DISABLE_AUTOUPDATER", "1"),
    ("ENABLE_PROMPT_CACHING_1H", "1"),
    // Names this workload in Claude Code's telemetry, where token-based runs carry no user.
    ("OTEL_RESOURCE_ATTRIBUTES", "job.name=spatial-agent"),
];

const OAUTH_TOKEN_VARIABLE: &str = "CLAUDE_CODE_OAUTH_TOKEN";

/// Variables that must not reach a claude process: a killed turn must never be re-run on resume.
const REMOVED_ENVIRONMENT: [&str; 1] = ["CLAUDE_CODE_RESUME_INTERRUPTED_TURN"];

/// Process-wide, so agent ids never repeat between launchers.
static SPAWNED: AtomicU64 = AtomicU64::new(0);

/// Whether a process may call the world tools.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolAccess {
    /// The world tools over the MCP endpoint at this URL, and nothing else.
    World(McpUrl),
    /// No tools at all.
    None,
}

/// What a claude process runs: its model, effort, system prompt and tools.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProcessProfile {
    pub model: ModelName,
    pub effort: Effort,
    pub system_prompt: SystemPrompt,
    pub tools: ToolAccess,
}

impl ProcessProfile {
    pub fn without_tools(profile: &ModelProfile) -> Self {
        Self {
            model: profile.model.clone(),
            effort: profile.effort,
            system_prompt: profile.system_prompt.clone(),
            tools: ToolAccess::None,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SpawnError {
    #[error("starting {claude}: {source}")]
    Start { claude: String, source: std::io::Error },
    #[error("{agent} did not report its MCP connection within {waited:?}")]
    Slow { agent: AgentId, waited: Duration },
    #[error("{agent} exited while starting")]
    ExitedEarly { agent: AgentId },
    #[error("{agent} cannot reach the world tools: MCP status {status}")]
    NoTools { agent: AgentId, status: String },
    #[error("{agent} has MCP servers it was not given: {status}")]
    UnexpectedTools { agent: AgentId, status: String },
    #[error("{agent} wrote to its input: {source}")]
    Input { agent: AgentId, source: std::io::Error },
    #[error("reading the Claude OAuth token file {path}: {source}")]
    TokenFile { path: PathBuf, source: std::io::Error },
    #[error("the Claude OAuth token file {path} is empty")]
    EmptyTokenFile { path: PathBuf },
    #[error("the Claude OAuth token file {path} can be read by others (mode {mode:o}); make it 600")]
    OpenTokenFile { path: PathBuf, mode: u32 },
}

/// A `claude setup-token` token. It never appears in logs or debug output.
struct OauthToken(String);

impl OauthToken {
    fn read(path: &Path) -> Result<Self, SpawnError> {
        let mode = std::fs::metadata(path).map_err(|source| SpawnError::TokenFile { path: path.to_owned(), source })?.permissions().mode();
        if mode & 0o077 != 0 {
            return Err(SpawnError::OpenTokenFile { path: path.to_owned(), mode: mode & 0o777 });
        }
        let text = std::fs::read_to_string(path).map_err(|source| SpawnError::TokenFile { path: path.to_owned(), source })?;
        let token = text.trim();
        if token.is_empty() {
            return Err(SpawnError::EmptyTokenFile { path: path.to_owned() });
        }
        Ok(Self(token.to_owned()))
    }
}

/// What a claude process says, or that it has gone.
#[derive(Debug)]
pub enum ProcessEvent {
    Output(OutputEvent),
    Garbled { line: String, error: ProtocolError },
    /// Its output closed: the process exited or is about to.
    Exited,
}

/// Starts claude processes of one profile with the lean flags, each with its own agent id, token and session.
pub struct Launcher {
    claude: ClaudeConfig,
    profile: ProcessProfile,
    registry: Arc<AgentRegistry>,
    startup: Duration,
    spawned: AtomicU64,
}

impl Launcher {
    pub fn new(claude: ClaudeConfig, profile: ProcessProfile, registry: Arc<AgentRegistry>, startup: Duration) -> Self {
        Self { claude, profile, registry, startup, spawned: AtomicU64::new(0) }
    }

    pub fn profile(&self) -> &ProcessProfile {
        &self.profile
    }

    /// How many processes this launcher has started.
    pub fn spawned(&self) -> u64 {
        self.spawned.load(Ordering::Relaxed)
    }

    /// The MCP server entry. The id and token are expanded from the child's environment, so they never appear on
    /// a command line.
    fn mcp_config(&self, url: &McpUrl) -> String {
        serde_json::json!({
            "mcpServers": {
                SERVER_NAME: {
                    "type": "http",
                    "url": url.0,
                    "headers": {AGENT_ID_HEADER: "${AGENT_ID}", "Authorization": "Bearer ${AGENT_TOKEN}"},
                    "timeout": self.claude.mcp_tool_timeout.as_millis(),
                }
            }
        })
        .to_string()
    }

    pub fn arguments(&self, session: SessionId) -> Vec<OsString> {
        let p = &self.profile;
        let mut arguments: Vec<String> = [
            "-p",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--model",
            &p.model.0,
            "--effort",
            p.effort.flag_value(),
            "--system-prompt",
            &p.system_prompt.0,
            "--tools",
            "",
            "--strict-mcp-config",
            "--permission-mode",
            "dontAsk",
            "--permission-prompts",
            "none",
            "--setting-sources",
            "",
            "--disable-slash-commands",
            "--session-id",
            &session.to_string(),
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        if let ToolAccess::World(url) = &p.tools {
            arguments.extend(["--mcp-config".to_owned(), self.mcp_config(url)]);
            arguments.extend(["--allowedTools".to_owned(), format!("mcp__{SERVER_NAME}")]);
        }
        arguments.into_iter().map(OsString::from).collect()
    }

    fn spawn(&self) -> Result<AgentProcess, SpawnError> {
        self.spawned.fetch_add(1, Ordering::Relaxed);
        let agent = AgentId(format!("agent-{}", SPAWNED.fetch_add(1, Ordering::Relaxed)));
        let session = SessionId::random();
        let oauth_token = match &self.claude.credentials {
            ClaudeCredentials::SignedInCli => None,
            ClaudeCredentials::OauthTokenFile(path) => Some(OauthToken::read(path)?),
        };
        let token = self.registry.register(agent.clone());
        let mut command = Command::new(&self.claude.executable);
        command
            .args(self.arguments(session))
            .current_dir(&self.claude.work_dir)
            .envs(LEAN_ENVIRONMENT)
            .env("AGENT_ID", &agent.0)
            .env("AGENT_TOKEN", token.expose())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        for name in REMOVED_ENVIRONMENT {
            command.env_remove(name);
        }
        if let Some(oauth_token) = &oauth_token {
            command.env(OAUTH_TOKEN_VARIABLE, &oauth_token.0);
        }
        let mut child = match command.spawn() {
            Ok(child) => child,
            Err(source) => {
                self.registry.unregister(&agent);
                return Err(SpawnError::Start { claude: self.claude.executable.display().to_string(), source });
            }
        };
        let stdin = child.stdin.take().expect("stdin is piped");
        let stdout = child.stdout.take().expect("stdout is piped");
        let stderr = child.stderr.take().expect("stderr is piped");
        let (events, receiver) = mpsc::unbounded_channel();
        tokio::spawn(read_output(stdout, events));
        let stderr_agent = agent.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                tracing::warn!(agent = %stderr_agent, "claude stderr: {line}");
            }
        });
        tracing::info!(%agent, %session, pid = child.id(), "claude started");
        Ok(AgentProcess { agent, session, child, stdin: Some(stdin), events: receiver, registry: self.registry.clone(), next_request: 0 })
    }

    /// A new process that reads its input and, when it has the world tools, whose MCP connection is up. It has no
    /// user message yet, so it costs no model call.
    pub async fn start(&self) -> Result<AgentProcess, SpawnError> {
        let mut process = self.spawn()?;
        let agent = process.agent.clone();
        let wants_world = matches!(self.profile.tools, ToolAccess::World(_));
        tokio::time::timeout(self.startup, async {
            loop {
                match process.mcp_status(wants_world).await? {
                    WorldConnection::Connected => return Ok(()),
                    WorldConnection::Pending => tokio::time::sleep(MCP_STATUS_POLL).await,
                    WorldConnection::Failed(status) => return Err(SpawnError::NoTools { agent: agent.clone(), status }),
                }
            }
        })
        .await
        .map_err(|_| SpawnError::Slow { agent: agent.clone(), waited: self.startup })??;
        Ok(process)
    }
}

const MCP_STATUS_POLL: Duration = Duration::from_millis(50);

/// The world server as `mcp_status` reports it; for a process without tools, connected means it answered.
enum WorldConnection {
    Connected,
    /// Still connecting.
    Pending,
    Failed(String),
}

async fn read_output(stdout: tokio::process::ChildStdout, events: mpsc::UnboundedSender<ProcessEvent>) {
    let mut lines = BufReader::new(stdout).lines();
    loop {
        let event = match lines.next_line().await {
            Ok(Some(line)) if line.trim().is_empty() => continue,
            Ok(Some(line)) => match OutputEvent::parse(&line) {
                Ok(event) => ProcessEvent::Output(event),
                Err(error) => ProcessEvent::Garbled { line, error },
            },
            Ok(None) => ProcessEvent::Exited,
            Err(error) => {
                tracing::error!(%error, "reading claude output");
                ProcessEvent::Exited
            }
        };
        let exited = matches!(event, ProcessEvent::Exited);
        if events.send(event).is_err() || exited {
            return;
        }
    }
}

/// A running claude process. Dropping it kills the process and withdraws its token.
pub struct AgentProcess {
    pub agent: AgentId,
    pub session: SessionId,
    child: Child,
    /// None once retired: closing it is how a session ends.
    stdin: Option<ChildStdin>,
    events: mpsc::UnboundedReceiver<ProcessEvent>,
    registry: Arc<AgentRegistry>,
    next_request: u64,
}

impl AgentProcess {
    pub async fn write(&mut self, message: &InputMessage) -> Result<(), SpawnError> {
        let line = message.to_line();
        let written = async {
            let stdin = self.stdin.as_mut().ok_or_else(|| std::io::Error::from(std::io::ErrorKind::BrokenPipe))?;
            stdin.write_all(line.as_bytes()).await?;
            stdin.flush().await
        };
        written.await.map_err(|source| SpawnError::Input { agent: self.agent.clone(), source })
    }

    pub async fn control(&mut self, request: ControlRequestBody) -> Result<RequestId, SpawnError> {
        self.next_request += 1;
        let id = RequestId(format!("{}-{}", self.agent, self.next_request));
        self.write(&InputMessage::control(id.clone(), request)).await?;
        Ok(id)
    }

    async fn mcp_status(&mut self, wants_world: bool) -> Result<WorldConnection, SpawnError> {
        let request_id = self.control(ControlRequestBody::McpStatus).await?;
        let reply = loop {
            match self.next_event().await {
                ProcessEvent::Output(OutputEvent::ControlResponse(response)) if response.response.request_id() == &request_id => {
                    break response.response;
                }
                ProcessEvent::Exited => return Err(SpawnError::ExitedEarly { agent: self.agent.clone() }),
                other => tracing::debug!(agent = %self.agent, ?other, "event before the MCP status"),
            }
        };
        let report = match reply {
            ControlResponseBody::Success { response, .. } => serde_json::from_value::<McpStatusReport>(response.clone())
                .map_err(|_| SpawnError::NoTools { agent: self.agent.clone(), status: response.to_string() })?,
            ControlResponseBody::Error { error, .. } => return Err(SpawnError::NoTools { agent: self.agent.clone(), status: error }),
        };
        if !wants_world {
            return match report.mcp_servers.is_empty() {
                true => Ok(WorldConnection::Connected),
                false => Err(SpawnError::UnexpectedTools { agent: self.agent.clone(), status: format!("{:?}", report.mcp_servers) }),
            };
        }
        Ok(match report.mcp_servers.iter().find(|s| s.name == SERVER_NAME) {
            Some(server) if server.is_connected() => WorldConnection::Connected,
            Some(server) if server.status == "pending" => WorldConnection::Pending,
            _ => WorldConnection::Failed(format!("{:?}", report.mcp_servers)),
        })
    }

    /// The next event; `Exited` once the output has closed, and from then on.
    pub async fn next_event(&mut self) -> ProcessEvent {
        self.events.recv().await.unwrap_or(ProcessEvent::Exited)
    }

    /// Whether it has already exited, without waiting.
    pub fn has_exited(&mut self) -> bool {
        !matches!(self.child.try_wait(), Ok(None))
    }

    pub fn terminate(&self) {
        if let Some(pid) = self.child.id() {
            if let Err(error) = kill(Pid::from_raw(pid as i32), Signal::SIGTERM) {
                tracing::error!(agent = %self.agent, %error, "SIGTERM failed");
            }
        }
    }

    pub fn kill(&mut self) {
        if let Err(error) = self.child.start_kill() {
            tracing::error!(agent = %self.agent, %error, "SIGKILL failed");
        }
    }

    /// Lets it finish: its input closes, which ends the session, and it is killed if still there after `grace`.
    pub fn retire(mut self, grace: Duration) {
        tokio::spawn(async move {
            self.registry.unregister(&self.agent);
            self.stdin.take();
            let agent = self.agent.clone();
            match tokio::time::timeout(grace, self.child.wait()).await {
                Ok(Ok(status)) => tracing::info!(%agent, %status, "claude retired"),
                Ok(Err(error)) => tracing::error!(%agent, %error, "waiting for a retired claude"),
                Err(_) => {
                    tracing::warn!(%agent, "retired claude still running after {grace:?}; killing it");
                    if let Err(error) = self.child.kill().await {
                        tracing::error!(%agent, %error, "killing a retired claude");
                    }
                }
            }
        });
    }
}

impl Drop for AgentProcess {
    fn drop(&mut self) {
        self.registry.unregister(&self.agent);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token_file(name: &str, text: &str, mode: u32) -> PathBuf {
        let path = std::env::temp_dir().join(format!("spatial-agent-token-{}-{name}", std::process::id()));
        std::fs::write(&path, text).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode)).unwrap();
        path
    }

    #[test]
    fn a_private_token_file_is_read_without_its_newline() {
        let path = token_file("private", "sk-ant-oat01-example\n", 0o600);
        assert_eq!(OauthToken::read(&path).unwrap().0, "sk-ant-oat01-example");
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn a_token_file_others_can_read_is_refused() {
        let path = token_file("open", "sk-ant-oat01-example", 0o644);
        assert!(matches!(OauthToken::read(&path), Err(SpawnError::OpenTokenFile { mode: 0o644, .. })));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn an_empty_token_file_is_refused() {
        let path = token_file("empty", "\n", 0o600);
        assert!(matches!(OauthToken::read(&path), Err(SpawnError::EmptyTokenFile { .. })));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn a_missing_token_file_is_refused() {
        let path = std::env::temp_dir().join("spatial-agent-token-that-does-not-exist");
        assert!(matches!(OauthToken::read(&path), Err(SpawnError::TokenFile { .. })));
    }
}
