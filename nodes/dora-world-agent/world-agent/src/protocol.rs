//! The stream-json protocol of `claude -p --input-format stream-json --output-format stream-json`: one JSON object
//! per line each way. Only what the supervisor acts on is typed; any other event is kept as raw JSON.

use std::fmt;

use serde::de::{self, Deserializer};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

/// The conversation a claude process holds; chosen by the supervisor with `--session-id`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SessionId(pub Uuid);

impl SessionId {
    pub fn random() -> Self {
        Self(Uuid::new_v4())
    }
}

impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Names one user message, so the result that answers it can be matched to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct MessageUuid(pub Uuid);

impl MessageUuid {
    pub fn random() -> Self {
        Self(Uuid::new_v4())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RequestId(pub String);

// Input: supervisor to claude.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Role {
    User,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct UserBody {
    role: Role,
    content: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "subtype", rename_all = "snake_case")]
pub enum ControlRequestBody {
    /// Ends the running turn; its result comes back with an aborted terminal reason.
    Interrupt {
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        cancel_queued: bool,
    },
    McpStatus,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputMessage {
    User {
        message: UserBody,
        parent_tool_use_id: Option<String>,
        uuid: MessageUuid,
    },
    ControlRequest { request_id: RequestId, request: ControlRequestBody },
}

impl InputMessage {
    pub fn user(uuid: MessageUuid, text: String) -> Self {
        InputMessage::User { message: UserBody { role: Role::User, content: text }, parent_tool_use_id: None, uuid }
    }

    pub fn control(request_id: RequestId, request: ControlRequestBody) -> Self {
        InputMessage::ControlRequest { request_id, request }
    }

    /// The message as one line of the input stream, newline included.
    pub fn to_line(&self) -> String {
        let mut line = serde_json::to_string(self).expect("input messages always serialise");
        line.push('\n');
        line
    }
}

// Output: claude to supervisor.

/// A feature the CLI announces in its init event.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Deserialize)]
#[serde(transparent)]
pub struct Capability(pub String);

pub const INTERRUPT_SEND_NOW: &str = "interrupt_send_now_v1";
pub const INTERRUPT_RECEIPT: &str = "interrupt_receipt_v1";

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct McpServerStatus {
    pub name: String,
    pub status: String,
}

impl McpServerStatus {
    pub fn is_connected(&self) -> bool {
        self.status == "connected"
    }
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct SystemInit {
    pub session_id: SessionId,
    pub model: String,
    pub tools: Vec<String>,
    pub mcp_servers: Vec<McpServerStatus>,
    #[serde(rename = "permissionMode")]
    pub permission_mode: String,
    #[serde(default)]
    pub capabilities: Vec<Capability>,
    pub claude_code_version: Option<String>,
}

impl SystemInit {
    pub fn has_capability(&self, name: &str) -> bool {
        self.capabilities.iter().any(|c| c.0 == name)
    }
}

/// What an API call failed with, as `api_retry` reports it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ApiErrorKind {
    AuthenticationFailed,
    RateLimit,
    Overloaded,
    Other(String),
}

impl<'de> Deserialize<'de> for ApiErrorKind {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(match String::deserialize(deserializer)?.as_str() {
            "authentication_failed" => ApiErrorKind::AuthenticationFailed,
            "rate_limit" => ApiErrorKind::RateLimit,
            "overloaded" => ApiErrorKind::Overloaded,
            other => ApiErrorKind::Other(other.to_owned()),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct ApiRetry {
    pub attempt: u32,
    pub max_retries: u32,
    pub error: Option<ApiErrorKind>,
    pub error_status: Option<u16>,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentDelta {
    TextDelta { text: String },
    InputJsonDelta { partial_json: String },
    #[serde(other)]
    Other,
}

/// One server-sent event of the model's reply, as `--include-partial-messages` forwards it.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PartialMessage {
    MessageStart,
    ContentBlockStart { index: u32 },
    ContentBlockDelta { index: u32, delta: ContentDelta },
    ContentBlockStop { index: u32 },
    MessageDelta,
    MessageStop,
    #[serde(other)]
    Other,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AssistantBlock {
    Text { text: String },
    ToolUse { id: String, name: String, input: Value },
    #[serde(other)]
    Other,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct AssistantBody {
    pub content: Vec<AssistantBlock>,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct AssistantMessage {
    pub message: AssistantBody,
    #[serde(default)]
    pub aborted: bool,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UserBlock {
    ToolResult {
        tool_use_id: String,
        content: Value,
        #[serde(default)]
        is_error: bool,
    },
    Text {
        text: String,
    },
    #[serde(other)]
    Other,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum UserContent {
    Text(String),
    Blocks(Vec<UserBlock>),
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct UserEchoBody {
    pub content: UserContent,
}

/// A user-role message in the output: tool results, or the CLI's own notes such as an interrupt marker.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct UserEcho {
    pub message: UserEchoBody,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResultSubtype {
    Success,
    ErrorDuringExecution,
    ErrorMaxTurns,
    Other(String),
}

impl<'de> Deserialize<'de> for ResultSubtype {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(match String::deserialize(deserializer)?.as_str() {
            "success" => ResultSubtype::Success,
            "error_during_execution" => ResultSubtype::ErrorDuringExecution,
            "error_max_turns" => ResultSubtype::ErrorMaxTurns,
            other => ResultSubtype::Other(other.to_owned()),
        })
    }
}

/// Why a turn ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TerminalReason {
    Completed,
    /// Interrupted while the model was writing.
    AbortedStreaming,
    /// Interrupted while a tool call was running.
    AbortedTools,
    Other(String),
}

impl TerminalReason {
    pub fn is_aborted(&self) -> bool {
        matches!(self, TerminalReason::AbortedStreaming | TerminalReason::AbortedTools)
    }
}

impl<'de> Deserialize<'de> for TerminalReason {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(match String::deserialize(deserializer)?.as_str() {
            "completed" => TerminalReason::Completed,
            "aborted_streaming" => TerminalReason::AbortedStreaming,
            "aborted_tools" => TerminalReason::AbortedTools,
            other => TerminalReason::Other(other.to_owned()),
        })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
    #[serde(default)]
    pub cache_read_input_tokens: u64,
    #[serde(default)]
    pub cache_creation_input_tokens: u64,
}

/// The end of a turn. Every result names the user messages it answers when they were sent with a uuid.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct TurnResult {
    pub subtype: ResultSubtype,
    pub is_error: bool,
    pub result: Option<String>,
    pub terminal_reason: Option<TerminalReason>,
    pub api_error_status: Option<u16>,
    pub session_id: SessionId,
    #[serde(default)]
    pub user_message_uuids: Vec<MessageUuid>,
    pub usage: Option<Usage>,
    pub total_cost_usd: Option<f64>,
    pub duration_ms: Option<u64>,
}

impl TurnResult {
    /// The API rejected the credentials: respawning will not help.
    pub fn is_auth_failure(&self) -> bool {
        matches!(self.api_error_status, Some(401 | 403))
    }
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(tag = "subtype", rename_all = "snake_case")]
pub enum ControlResponseBody {
    Success {
        request_id: RequestId,
        #[serde(default)]
        response: Value,
    },
    Error {
        request_id: RequestId,
        error: String,
    },
}

impl ControlResponseBody {
    pub fn request_id(&self) -> &RequestId {
        match self {
            ControlResponseBody::Success { request_id, .. } | ControlResponseBody::Error { request_id, .. } => request_id,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct ControlResponse {
    pub response: ControlResponseBody,
}

/// The answer to `mcp_status`.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpStatusReport {
    pub mcp_servers: Vec<McpServerStatus>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum OutputEvent {
    SystemInit(Box<SystemInit>),
    ApiRetry(ApiRetry),
    PartialMessage(PartialMessage),
    Assistant(AssistantMessage),
    User(UserEcho),
    Result(Box<TurnResult>),
    ControlResponse(ControlResponse),
    /// Any event the supervisor does not act on, kept as it came.
    Unrecognised(Value),
}

#[derive(Debug, thiserror::Error)]
pub enum ProtocolError {
    #[error("output line is not JSON: {0}")]
    NotJson(#[source] serde_json::Error),
    #[error("output line is not a JSON object with a type")]
    Untyped,
    #[error("malformed {event} event: {source}")]
    Malformed { event: &'static str, source: serde_json::Error },
}

fn typed<T: for<'de> Deserialize<'de>>(event: &'static str, value: Value) -> Result<T, ProtocolError> {
    serde_json::from_value(value).map_err(|source| ProtocolError::Malformed { event, source })
}

impl OutputEvent {
    pub fn parse(line: &str) -> Result<OutputEvent, ProtocolError> {
        let value: Value = serde_json::from_str(line).map_err(ProtocolError::NotJson)?;
        let kind = value.get("type").and_then(Value::as_str).ok_or(ProtocolError::Untyped)?.to_owned();
        let subtype = value.get("subtype").and_then(Value::as_str).map(str::to_owned);
        Ok(match (kind.as_str(), subtype.as_deref()) {
            ("system", Some("init")) => OutputEvent::SystemInit(Box::new(typed("system init", value)?)),
            ("system", Some("api_retry")) => OutputEvent::ApiRetry(typed("api_retry", value)?),
            ("stream_event", _) => {
                let event = value.get("event").cloned().ok_or_else(|| ProtocolError::Malformed {
                    event: "stream_event",
                    source: de::Error::missing_field("event"),
                })?;
                OutputEvent::PartialMessage(typed("stream_event", event)?)
            }
            ("assistant", _) => OutputEvent::Assistant(typed("assistant", value)?),
            ("user", _) => OutputEvent::User(typed("user", value)?),
            ("result", _) => OutputEvent::Result(Box::new(typed("result", value)?)),
            ("control_response", _) => OutputEvent::ControlResponse(typed("control_response", value)?),
            _ => OutputEvent::Unrecognised(value),
        })
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn fixture(name: &str) -> OutputEvent {
        let path = format!("{}/tests/fixtures/stream-json/{name}.json", env!("CARGO_MANIFEST_DIR"));
        let line = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        OutputEvent::parse(line.trim_end()).unwrap_or_else(|e| panic!("{name}: {e}"))
    }

    #[test]
    fn the_init_event_lists_tools_permission_mode_and_capabilities() {
        let OutputEvent::SystemInit(init) = fixture("system_init") else { panic!("not an init event") };
        assert_eq!(init.permission_mode, "dontAsk");
        assert_eq!(init.model, "claude-haiku-5-5");
        assert!(init.tools.contains(&"mcp__world__send_goal".to_owned()));
        assert!(init.mcp_servers.iter().any(|s| s.name == "world" && s.is_connected()));
        assert!(init.has_capability(INTERRUPT_SEND_NOW));
        assert!(init.has_capability(INTERRUPT_RECEIPT));
    }

    #[test]
    fn a_tool_call_is_read_from_the_assistant_message() {
        let OutputEvent::Assistant(message) = fixture("assistant_tool_use") else { panic!("not an assistant event") };
        let [AssistantBlock::ToolUse { name, input, .. }] = &message.message.content[..] else {
            panic!("{message:?}")
        };
        assert_eq!(name, "mcp__world__find");
        assert_eq!(input, &json!({"query": "kitchen table"}));
    }

    #[test]
    fn a_tool_result_is_read_from_the_user_echo() {
        let OutputEvent::User(echo) = fixture("user_tool_result") else { panic!("not a user event") };
        let UserContent::Blocks(blocks) = echo.message.content else { panic!("no blocks") };
        assert!(matches!(&blocks[..], [UserBlock::ToolResult { is_error: false, .. }]));
    }

    #[test]
    fn partial_messages_carry_text_deltas_and_tool_starts() {
        assert_eq!(fixture("stream_message_start"), OutputEvent::PartialMessage(PartialMessage::MessageStart));
        assert!(matches!(
            fixture("stream_text_delta"),
            OutputEvent::PartialMessage(PartialMessage::ContentBlockDelta { delta: ContentDelta::TextDelta { .. }, .. })
        ));
        assert_eq!(fixture("stream_tool_use_start"), OutputEvent::PartialMessage(PartialMessage::ContentBlockStart { index: 0 }));
    }

    #[test]
    fn a_completed_result_names_the_message_it_answers() {
        let OutputEvent::Result(result) = fixture("result_success") else { panic!("not a result") };
        assert_eq!(result.subtype, ResultSubtype::Success);
        assert_eq!(result.terminal_reason, Some(TerminalReason::Completed));
        assert!(!result.is_error);
        assert_eq!(result.user_message_uuids.len(), 1);
        assert!(result.result.unwrap().starts_with("The robot is heading to the kitchen table"));
        assert_eq!(result.usage.unwrap().cache_read_input_tokens, 4412);
    }

    #[test]
    fn an_interrupted_result_is_an_aborted_error() {
        let OutputEvent::Result(result) = fixture("result_interrupted") else { panic!("not a result") };
        assert_eq!(result.subtype, ResultSubtype::ErrorDuringExecution);
        assert!(result.is_error && !result.is_auth_failure());
        assert!(result.terminal_reason.unwrap().is_aborted());
    }

    #[test]
    fn the_interrupt_receipt_is_a_successful_control_response() {
        let OutputEvent::ControlResponse(response) = fixture("control_response_interrupt") else { panic!() };
        assert_eq!(response.response.request_id(), &RequestId("int-1".to_owned()));
        assert!(matches!(response.response, ControlResponseBody::Success { .. }));
    }

    #[test]
    fn an_api_retry_names_its_error() {
        let OutputEvent::ApiRetry(retry) = fixture("system_api_retry") else { panic!("not an api_retry") };
        assert_eq!(retry.error, Some(ApiErrorKind::AuthenticationFailed));
        assert_eq!((retry.attempt, retry.max_retries, retry.error_status), (1, 10, Some(401)));
    }

    #[test]
    fn events_the_supervisor_does_not_act_on_are_kept_raw() {
        for name in ["rate_limit_event", "command_lifecycle", "system_status"] {
            let OutputEvent::Unrecognised(value) = fixture(name) else { panic!("{name} was typed") };
            assert!(value.get("type").is_some(), "{name}");
        }
    }

    #[test]
    fn a_known_event_with_missing_fields_is_a_protocol_error() {
        let error = OutputEvent::parse(r#"{"type":"result","subtype":"success"}"#).unwrap_err();
        assert!(matches!(error, ProtocolError::Malformed { event: "result", .. }), "{error}");
        assert!(matches!(OutputEvent::parse("not json"), Err(ProtocolError::NotJson(_))));
        assert!(matches!(OutputEvent::parse(r#"{"no":"type"}"#), Err(ProtocolError::Untyped)));
    }

    #[test]
    fn a_user_message_is_written_with_its_uuid() {
        let uuid = MessageUuid(Uuid::nil());
        let line = InputMessage::user(uuid, "Where is the mug?".to_owned()).to_line();
        assert!(line.ends_with('\n'));
        let value: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(
            value,
            json!({
                "type": "user",
                "message": {"role": "user", "content": "Where is the mug?"},
                "parent_tool_use_id": null,
                "uuid": "00000000-0000-0000-0000-000000000000",
            })
        );
    }

    #[test]
    fn control_requests_are_written_with_their_subtype() {
        let interrupt = InputMessage::control(RequestId("r1".to_owned()), ControlRequestBody::Interrupt { cancel_queued: false });
        assert_eq!(
            serde_json::from_str::<Value>(&interrupt.to_line()).unwrap(),
            json!({"type": "control_request", "request_id": "r1", "request": {"subtype": "interrupt"}})
        );
        let status = InputMessage::control(RequestId("r2".to_owned()), ControlRequestBody::McpStatus);
        assert_eq!(
            serde_json::from_str::<Value>(&status.to_line()).unwrap(),
            json!({"type": "control_request", "request_id": "r2", "request": {"subtype": "mcp_status"}})
        );
    }
}
