//! `ModelClient` on the Claude Code CLI: one `claude -p` process per model call, over stream-json.
//!
//! Each client keeps one process of its model started and connected ahead of need. A call takes that warm process,
//! a replacement starts at once, and the call writes its one user message, waits for the result that names it and
//! retires the process, so every call is a fresh session. A call that goes silent is interrupted with a control
//! request, then the process gets SIGTERM, then SIGKILL. Refused credentials, or a process that starts with another
//! setup than it was launched with, take the client down: every later call fails at once with the reason.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::task::JoinHandle;
use tokio::time::Instant;

use crate::config::{ModelName, WatchdogConfig};
use crate::identity::AgentRegistry;
use crate::mcp::{SERVER_NAME, TOOL_NAMES};
use crate::model::{Duty, ModelClient, ModelReply, Prompt, Spend};
use crate::process::{AgentProcess, Launcher, ProcessEvent, SpawnError, ToolAccess};
use crate::protocol::{
    ApiErrorKind, ControlRequestBody, InputMessage, MessageUuid, OutputEvent, ResultSubtype, SessionId, SystemInit,
    TurnResult, INTERRUPT_RECEIPT, INTERRUPT_SEND_NOW,
};

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum DownReason {
    #[error("the API refused the credentials")]
    AuthenticationFailed,
    #[error("claude started with the wrong setup: {0}")]
    Misconfigured(String),
}

#[derive(Debug, thiserror::Error)]
pub enum AskError {
    #[error("{model} is down: {reason}")]
    Down { model: ModelName, reason: DownReason },
    #[error("no claude process: {0}")]
    Spawn(#[from] SpawnError),
    #[error("the call went silent and was stopped")]
    TimedOut,
    #[error("the claude process exited during the call")]
    Exited,
    #[error("the call failed ({subtype:?}): {message}")]
    Failed { subtype: ResultSubtype, message: String },
    #[error("claude broke the stream-json protocol: {0}")]
    Protocol(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Escalation {
    Calm,
    Interrupted(Instant),
    Terminated(Instant),
    Killed,
}

fn expected_tools(tools: &ToolAccess) -> Vec<String> {
    let mut names: Vec<String> = match tools {
        ToolAccess::World(_) => TOOL_NAMES.iter().map(|t| format!("mcp__{SERVER_NAME}__{t}")).collect(),
        ToolAccess::None => Vec::new(),
    };
    names.sort();
    names
}

/// Why an init event shows a process that is not set up as launched, if it is not.
fn init_problem(init: &SystemInit, session: SessionId, tools: &ToolAccess) -> Option<String> {
    let mut listed = init.tools.clone();
    listed.sort();
    let world_connected = init.mcp_servers.iter().any(|s| s.name == SERVER_NAME && s.is_connected());
    if init.session_id != session {
        Some(format!("session {} instead of {session}", init.session_id))
    } else if init.permission_mode != "dontAsk" {
        Some(format!("permission mode {}", init.permission_mode))
    } else if listed != expected_tools(tools) {
        Some(format!("tools {listed:?}"))
    } else if matches!(tools, ToolAccess::World(_)) && !world_connected {
        Some(format!("MCP servers {:?}", init.mcp_servers))
    } else if !(init.has_capability(INTERRUPT_SEND_NOW) && init.has_capability(INTERRUPT_RECEIPT)) {
        Some("no interrupt capabilities".to_owned())
    } else {
        None
    }
}

/// How one call ended, before the client decides whether that takes it down.
enum CallEnd {
    Reply(ModelReply),
    Error(AskError),
    Down(DownReason),
}

/// One model call on one process, from the user message to the result that names it.
struct Call<'a> {
    process: &'a mut AgentProcess,
    tools: &'a ToolAccess,
    watchdog: WatchdogConfig,
    uuid: MessageUuid,
    last_event: Instant,
    escalation: Escalation,
}

impl Call<'_> {
    async fn run(mut self, prompt: Prompt) -> CallEnd {
        if let Err(error) = self.process.write(&InputMessage::user(self.uuid, prompt.0)).await {
            return CallEnd::Error(error.into());
        }
        let tick = (self.watchdog.silent_turn / 5).min(Duration::from_millis(250));
        let mut ticker = tokio::time::interval(tick);
        loop {
            tokio::select! {
                event = self.process.next_event() => {
                    self.last_event = Instant::now();
                    if let Some(end) = self.handle(event) {
                        return end;
                    }
                }
                _ = ticker.tick() => {
                    if let Err(error) = self.watchdog().await {
                        return CallEnd::Error(error.into());
                    }
                }
            }
        }
    }

    fn handle(&mut self, event: ProcessEvent) -> Option<CallEnd> {
        match event {
            ProcessEvent::Output(OutputEvent::SystemInit(init)) => {
                init_problem(&init, self.process.session, self.tools).map(|problem| CallEnd::Down(DownReason::Misconfigured(problem)))
            }
            ProcessEvent::Output(OutputEvent::ApiRetry(retry)) => {
                tracing::warn!(agent = %self.process.agent, ?retry, "API call retried");
                (retry.error == Some(ApiErrorKind::AuthenticationFailed)).then_some(CallEnd::Down(DownReason::AuthenticationFailed))
            }
            ProcessEvent::Output(OutputEvent::Result(result)) => Some(self.finish(*result)),
            ProcessEvent::Output(_) => None,
            ProcessEvent::Garbled { line, error } => {
                tracing::error!(agent = %self.process.agent, %error, line, "unreadable output");
                Some(CallEnd::Error(AskError::Protocol(error.to_string())))
            }
            ProcessEvent::Exited => Some(CallEnd::Error(match self.escalation {
                Escalation::Calm => AskError::Exited,
                _ => AskError::TimedOut,
            })),
        }
    }

    fn finish(&self, result: TurnResult) -> CallEnd {
        if result.is_auth_failure() {
            return CallEnd::Down(DownReason::AuthenticationFailed);
        }
        if !result.user_message_uuids.contains(&self.uuid) {
            return CallEnd::Error(AskError::Protocol(format!("a result for {:?}, not the call's message", result.user_message_uuids)));
        }
        if result.terminal_reason.as_ref().is_some_and(|r| r.is_aborted()) {
            return CallEnd::Error(AskError::TimedOut);
        }
        if result.is_error {
            let message = result.result.unwrap_or_else(|| "claude gave no message".to_owned());
            return CallEnd::Error(AskError::Failed { subtype: result.subtype, message });
        }
        let Some(text) = result.result else {
            return CallEnd::Error(AskError::Protocol("a successful result with no text".to_owned()));
        };
        let usage = result.usage.unwrap_or_default();
        CallEnd::Reply(ModelReply {
            text,
            spend: Spend { input_tokens: usage.input_tokens, output_tokens: usage.output_tokens, cost_usd: result.total_cost_usd },
        })
    }

    async fn watchdog(&mut self) -> Result<(), SpawnError> {
        let watchdog = self.watchdog;
        let agent = self.process.agent.clone();
        match self.escalation {
            Escalation::Calm if self.last_event.elapsed() > watchdog.silent_turn => {
                tracing::warn!(%agent, "call silent for {:?}; interrupting", watchdog.silent_turn);
                self.escalation = Escalation::Interrupted(Instant::now());
                self.process.control(ControlRequestBody::Interrupt { cancel_queued: true }).await?;
            }
            Escalation::Interrupted(at) if at.elapsed() > watchdog.interrupt_grace => {
                tracing::warn!(%agent, "interrupt not honoured; SIGTERM");
                self.process.terminate();
                self.escalation = Escalation::Terminated(Instant::now());
            }
            Escalation::Terminated(at) if at.elapsed() > watchdog.terminate_grace => {
                tracing::warn!(%agent, "still running after SIGTERM; SIGKILL");
                self.process.kill();
                self.escalation = Escalation::Killed;
            }
            _ => {}
        }
        Ok(())
    }
}

type WarmStart = JoinHandle<Result<AgentProcess, SpawnError>>;

/// One configured model on the Claude Code CLI, with one warm process kept ready.
pub struct ClaudeCodeClient {
    launcher: Arc<Launcher>,
    registry: Arc<AgentRegistry>,
    watchdog: WatchdogConfig,
    warm: Mutex<Option<WarmStart>>,
    down: Mutex<Option<DownReason>>,
}

impl ClaudeCodeClient {
    /// The client, with its first warm process starting.
    pub fn start(launcher: Launcher, registry: Arc<AgentRegistry>, watchdog: WatchdogConfig) -> Self {
        let client = Self { launcher: Arc::new(launcher), registry, watchdog, warm: Mutex::new(None), down: Mutex::new(None) };
        client.refill();
        client
    }

    fn refill(&self) {
        let launcher = self.launcher.clone();
        let previous = self.warm.lock().expect("warm lock").replace(tokio::spawn(async move { launcher.start().await }));
        if let Some(previous) = previous {
            previous.abort();
        }
    }

    /// How many processes this client has started, the warm one included.
    pub fn spawned(&self) -> u64 {
        self.launcher.spawned()
    }

    pub fn down(&self) -> Option<DownReason> {
        self.down.lock().expect("down lock").clone()
    }

    fn go_down(&self, reason: DownReason) {
        tracing::error!(model = %self.model(), %reason, "model client down");
        *self.down.lock().expect("down lock") = Some(reason);
        if let Some(warm) = self.warm.lock().expect("warm lock").take() {
            warm.abort();
        }
    }

    /// The warm process; its replacement starts before this one is awaited.
    async fn take_warm(&self) -> Result<AgentProcess, AskError> {
        let pending = self.warm.lock().expect("warm lock").take().expect("an up client always holds a warm process");
        self.refill();
        let mut process = pending.await.expect("a warm start does not panic")?;
        if process.has_exited() {
            return Err(SpawnError::ExitedEarly { agent: process.agent.clone() }.into());
        }
        Ok(process)
    }

    /// Ends the client's processes: the warm one is stopped and killed.
    pub fn stop(&self) {
        if let Some(warm) = self.warm.lock().expect("warm lock").take() {
            warm.abort();
        }
    }
}

impl ModelClient for ClaudeCodeClient {
    type Error = AskError;

    fn model(&self) -> &ModelName {
        &self.launcher.profile().model
    }

    async fn ask(&self, duty: Duty, prompt: Prompt) -> Result<ModelReply, AskError> {
        let down = |reason| AskError::Down { model: self.model().clone(), reason };
        if let Some(reason) = self.down() {
            return Err(down(reason));
        }
        let mut process = self.take_warm().await?;
        self.registry.assign(&process.agent, duty);
        tracing::info!(agent = %process.agent, session = %process.session, model = %self.model(), ?duty, "model call");
        let call = Call {
            process: &mut process,
            tools: &self.launcher.profile().tools,
            watchdog: self.watchdog,
            uuid: MessageUuid::random(),
            last_event: Instant::now(),
            escalation: Escalation::Calm,
        };
        let end = call.run(prompt).await;
        process.retire(self.watchdog.terminate_grace);
        match end {
            CallEnd::Reply(reply) => Ok(reply),
            CallEnd::Error(error) => Err(error),
            CallEnd::Down(reason) => {
                self.go_down(reason.clone());
                Err(down(reason))
            }
        }
    }
}

impl Drop for ClaudeCodeClient {
    fn drop(&mut self) {
        self.stop();
    }
}
