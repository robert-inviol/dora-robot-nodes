//! Settings and a work directory for driving `fake-claude`.

#![allow(dead_code)]

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::time::Duration;

use perception_wire::navigation::{NonNegative, UnixNanos};
use world_agent::config::{
    AnswererConfig, AnswererRole, AnsweringConfig, ClaudeConfig, ClaudeCredentials, CompositionChange, Effort, GoalDefaults, McpConfig,
    ModelName, ModelProfile, RoomNamerConfig, SpatialConfig, SystemPrompt, WatchdogConfig, WorldStoreSettings,
};

pub const NOW: UnixNanos = UnixNanos(1_794_610_000_000_000_000);
pub const FAST: &str = "fast-model";
pub const SLOW_A: &str = "slow-a";
pub const SLOW_B: &str = "slow-b";
pub const RECONCILER: &str = "reconciler";
pub const NAMER: &str = "namer";

pub struct WorkDir(pub PathBuf);

impl WorkDir {
    pub fn new() -> Self {
        Self(std::env::temp_dir().join(format!("spatial-agent-test-{}", uuid::Uuid::new_v4())))
    }
}

impl Drop for WorkDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub fn claude(work_dir: &Path) -> ClaudeConfig {
    ClaudeConfig {
        executable: PathBuf::from(env!("CARGO_BIN_EXE_fake-claude")),
        work_dir: work_dir.to_owned(),
        mcp_tool_timeout: Duration::from_secs(5),
        credentials: ClaudeCredentials::SignedInCli,
    }
}

pub fn watchdog() -> WatchdogConfig {
    WatchdogConfig {
        silent_turn: Duration::from_millis(300),
        interrupt_grace: Duration::from_millis(300),
        terminate_grace: Duration::from_millis(300),
        startup: Duration::from_secs(10),
    }
}

fn model(name: &str) -> ModelName {
    ModelName(name.to_owned())
}

pub fn config(work_dir: &Path, deadline: Duration) -> SpatialConfig {
    let answerer = |name: &str, role| AnswererConfig { model: model(name), effort: Effort::Low, role };
    SpatialConfig {
        claude: claude(work_dir),
        answering: AnsweringConfig {
            system_prompt: SystemPrompt("You answer questions.".to_owned()),
            deadline,
            answerers: vec![
                answerer(FAST, AnswererRole::Fast),
                answerer(SLOW_A, AnswererRole::Deliberate),
                answerer(SLOW_B, AnswererRole::Deliberate),
            ],
        },
        reconciler: ModelProfile {
            model: model(RECONCILER),
            effort: Effort::Low,
            system_prompt: SystemPrompt("You reconcile. [fake:reply={\"decision\": \"keep\"}]".to_owned()),
        },
        room_namer: RoomNamerConfig {
            model: model(NAMER),
            effort: Effort::Low,
            system_prompt: SystemPrompt("You name rooms. [fake:reply=living room]".to_owned()),
            rename_after_change: CompositionChange::new(0.5).unwrap(),
        },
        watchdog: watchdog(),
        goals: GoalDefaults {
            standoff_m: NonNegative::new("standoff_m", 0.5).unwrap(),
            max_speed_m_s: NonNegative::new("max_speed_m_s", 0.3).unwrap(),
            deadline: Duration::from_secs(60),
        },
        mcp: McpConfig { port: 0, sql_row_limit: NonZeroUsize::new(10).unwrap() },
        world_store: WorldStoreSettings { path: work_dir.with_extension("sqlite"), backup: None },
    }
}
