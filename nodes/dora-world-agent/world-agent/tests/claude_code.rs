//! One `ClaudeCodeClient` driving `fake-claude`: the launch, the warm process, the watchdog and going down.

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::{claude, watchdog, WorkDir};
use perception_wire::navigation::{QuestionId, RoomId};
use serde_json::Value;
use world_agent::claude_code::{AskError, ClaudeCodeClient, DownReason};
use world_agent::config::{AnswererRole, Effort, ModelName, SystemPrompt};
use world_agent::identity::AgentRegistry;
use world_agent::mcp::McpUrl;
use world_agent::model::{AnswererId, Duty, ModelClient, ModelReply, Prompt};
use world_agent::process::{Launcher, ProcessProfile, ToolAccess};

const MODEL: &str = "test-model";
const ANSWER: Duty = Duty::Answer { question: QuestionId(1), answerer: AnswererId(0), role: AnswererRole::Fast };

struct Harness {
    client: ClaudeCodeClient,
    work_dir: WorkDir,
}

impl Harness {
    fn start(system_prompt: &str, tools: ToolAccess) -> Self {
        let work_dir = WorkDir::new();
        std::fs::create_dir_all(&work_dir.0).unwrap();
        let profile = ProcessProfile {
            model: ModelName(MODEL.to_owned()),
            effort: Effort::Low,
            system_prompt: SystemPrompt(system_prompt.to_owned()),
            tools,
        };
        let registry = Arc::new(AgentRegistry::default());
        let launcher = Launcher::new(claude(&work_dir.0), profile, registry.clone(), watchdog().startup);
        Self { client: ClaudeCodeClient::start(launcher, registry, watchdog()), work_dir }
    }

    fn with_world_tools() -> Self {
        Self::start("You are a test.", ToolAccess::World(McpUrl("http://127.0.0.1:9/mcp".to_owned())))
    }

    async fn ask(&self, step: &str) -> Result<ModelReply, AskError> {
        let prompt = Prompt(format!("Hello.\nfake {MODEL} {step}"));
        tokio::time::timeout(Duration::from_secs(20), self.client.ask(ANSWER, prompt)).await.expect("the call ends")
    }

    async fn argv(&self) -> Value {
        serde_json::from_str(&self.ask("argv").await.unwrap().text).unwrap()
    }
}

fn arguments(report: &Value) -> Vec<&str> {
    report["args"].as_array().unwrap().iter().map(|a| a.as_str().unwrap()).collect()
}

fn after<'a>(args: &[&'a str], flag: &str) -> &'a str {
    args[args.iter().position(|a| *a == flag).unwrap_or_else(|| panic!("no {flag}")) + 1]
}

#[tokio::test]
async fn an_answerer_starts_with_the_lean_flags_its_model_and_the_world_tools() {
    let harness = Harness::with_world_tools();
    let report = harness.argv().await;
    let args = arguments(&report);
    assert_eq!(&args[..6], ["-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose"]);
    for flag in ["--include-partial-messages", "--strict-mcp-config", "--disable-slash-commands"] {
        assert!(args.contains(&flag), "{flag}");
    }
    assert_eq!(after(&args, "--model"), MODEL);
    assert_eq!(after(&args, "--tools"), "");
    assert_eq!(after(&args, "--setting-sources"), "");
    assert_eq!(after(&args, "--permission-mode"), "dontAsk");
    assert_eq!(after(&args, "--allowedTools"), "mcp__world");
    assert_eq!(after(&args, "--effort"), "low");
    let mcp: Value = serde_json::from_str(after(&args, "--mcp-config")).unwrap();
    let world = &mcp["mcpServers"]["world"];
    assert_eq!(world["type"], "http");
    assert_eq!(world["headers"]["x-agent-id"], "${AGENT_ID}");
    assert_eq!(world["headers"]["Authorization"], "Bearer ${AGENT_TOKEN}");
    assert_eq!(world["timeout"], 5000);
    assert_eq!(report["env"]["ENABLE_TOOL_SEARCH"], "false");
    assert_eq!(report["env"]["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"], "1");
    assert_eq!(report["env"]["CLAUDE_CODE_RESUME_INTERRUPTED_TURN"], Value::Null);
    assert_eq!(report["has_token"], true);
    assert_eq!(report["cwd"], harness.work_dir.0.display().to_string());
}

#[tokio::test]
async fn a_model_without_tools_gets_no_mcp_server() {
    let harness = Harness::start("You reconcile.", ToolAccess::None);
    let report = harness.argv().await;
    let args = arguments(&report);
    assert!(!args.contains(&"--mcp-config"), "{args:?}");
    assert!(!args.contains(&"--allowedTools"), "{args:?}");
    assert_eq!(after(&args, "--tools"), "");
    assert!(args.contains(&"--strict-mcp-config"));
}

async fn until_spawned(client: &ClaudeCodeClient, count: u64) {
    tokio::time::timeout(Duration::from_millis(400), async {
        while client.spawned() < count {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("{} processes started, not {count}", client.spawned()));
}

#[tokio::test]
async fn every_call_takes_the_warm_process_and_a_replacement_starts_at_once() {
    let harness = Arc::new(Harness::with_world_tools());
    until_spawned(&harness.client, 1).await;
    let first = harness.argv().await;
    let second = harness.argv().await;
    assert_ne!(first["env"]["AGENT_ID"], second["env"]["AGENT_ID"]);
    let (first_args, second_args) = (arguments(&first), arguments(&second));
    assert_ne!(after(&first_args, "--session-id"), after(&second_args, "--session-id"));

    let spawned = harness.client.spawned();
    let running = {
        let harness = harness.clone();
        tokio::spawn(async move { harness.ask("wait 1000\nfake test-model say done").await })
    };
    until_spawned(&harness.client, spawned + 1).await;
    assert!(!running.is_finished(), "the replacement started only after the call");
    assert_eq!(running.await.unwrap().unwrap().text, "done");
}

#[tokio::test]
async fn calls_run_side_by_side_on_their_own_processes() {
    let harness = Arc::new(Harness::with_world_tools());
    let slow = {
        let harness = harness.clone();
        tokio::spawn(async move { harness.ask("wait 200\nfake test-model say slow").await })
    };
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(harness.ask("say quick").await.unwrap().text, "quick");
    assert_eq!(slow.await.unwrap().unwrap().text, "slow");
}

#[tokio::test]
async fn a_silent_call_that_honours_the_interrupt_times_out() {
    let harness = Harness::with_world_tools();
    assert!(matches!(harness.ask("hang").await, Err(AskError::TimedOut)));
    assert_eq!(harness.ask("say still here").await.unwrap().text, "still here");
}

#[tokio::test]
async fn a_silent_call_that_ignores_the_interrupt_is_terminated() {
    let harness = Harness::with_world_tools();
    assert!(matches!(harness.ask("deaf").await, Err(AskError::TimedOut)));
    assert_eq!(harness.ask("say still here").await.unwrap().text, "still here");
}

#[tokio::test]
async fn a_crash_fails_only_its_own_call() {
    let harness = Harness::with_world_tools();
    assert!(matches!(harness.ask("crash").await, Err(AskError::Exited)));
    assert_eq!(harness.ask("say fine").await.unwrap().text, "fine");
}

#[tokio::test]
async fn refused_credentials_take_the_client_down() {
    let harness = Harness::with_world_tools();
    let error = harness.ask("auth").await.unwrap_err();
    assert!(matches!(error, AskError::Down { reason: DownReason::AuthenticationFailed, .. }), "{error}");
    let error = harness.ask("say hello").await.unwrap_err();
    assert!(matches!(error, AskError::Down { reason: DownReason::AuthenticationFailed, .. }), "{error}");
    assert_eq!(harness.client.down(), Some(DownReason::AuthenticationFailed));
}

#[tokio::test]
async fn a_process_started_with_the_wrong_permission_mode_takes_the_client_down() {
    let harness = Harness::start("You are a test. [fake:permission=default]", ToolAccess::None);
    let error = harness.ask("say hello").await.unwrap_err();
    assert!(
        matches!(&error, AskError::Down { reason: DownReason::Misconfigured(problem), .. } if problem == "permission mode default"),
        "{error}"
    );
}

#[tokio::test]
async fn a_tool_call_from_a_process_without_tools_fails_the_call() {
    let harness = Harness::start("You name rooms.", ToolAccess::None);
    let prompt = Prompt(format!("fake {MODEL} call find {{}}"));
    let error = harness.client.ask(Duty::NameRoom { room: RoomId(1) }, prompt).await.unwrap_err();
    assert!(error.to_string().contains("this process has no MCP server"), "{error}");
}
