//! The settings file, in TOML. `spatial-agent.example.toml` is one to start from.

use std::fmt;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::time::Duration;

use perception_wire::navigation::NonNegative;
use serde::{Deserialize, Deserializer};

use crate::goals::Approach;

/// A model name or alias as `claude --model` takes it.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Deserialize)]
#[serde(transparent)]
pub struct ModelName(pub String);

impl fmt::Display for ModelName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Effort {
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

impl Effort {
    pub fn flag_value(self) -> &'static str {
        match self {
            Effort::Low => "low",
            Effort::Medium => "medium",
            Effort::High => "high",
            Effort::Xhigh => "xhigh",
            Effort::Max => "max",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(transparent)]
pub struct SystemPrompt(pub String);

fn seconds<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Duration, D::Error> {
    let value = f64::deserialize(deserializer)?;
    Duration::try_from_secs_f64(value).map_err(|e| serde::de::Error::custom(format!("{value} seconds: {e}")))
}

fn metres<'de, D: Deserializer<'de>>(deserializer: D) -> Result<NonNegative, D::Error> {
    let value = f32::deserialize(deserializer)?;
    NonNegative::new("distance or speed", value).map_err(serde::de::Error::custom)
}

/// How every claude process is started, whatever model it runs.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClaudeConfig {
    /// The claude executable; a bare name is looked up on PATH.
    pub executable: PathBuf,
    /// The empty directory every claude process runs in.
    pub work_dir: PathBuf,
    /// How long claude waits for one tool call before giving up on it.
    #[serde(rename = "mcp_tool_timeout_s", deserialize_with = "seconds")]
    pub mcp_tool_timeout: Duration,
}

/// Whether an answerer's goals move the robot at once or wait for the reconciler.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AnswererRole {
    /// Answers first; its answer goes out at once and its goals are sent at once.
    Fast,
    /// Answers later; its goals are proposals that only the reconciler can turn into nav goals.
    Deliberate,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnswererConfig {
    pub model: ModelName,
    pub effort: Effort,
    pub role: AnswererRole,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnsweringConfig {
    /// The system prompt of every answerer.
    pub system_prompt: SystemPrompt,
    /// After this long a question gets its final message, whatever is still running.
    #[serde(rename = "deadline_s", deserialize_with = "seconds")]
    pub deadline: Duration,
    /// Every model that answers each question, in parallel. Exactly one is fast.
    pub answerers: Vec<AnswererConfig>,
}

/// A model call with no tools: the reconciler and the room namer.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelProfile {
    pub model: ModelName,
    pub effort: Effort,
    pub system_prompt: SystemPrompt,
}

/// The share of a room's objects that may change before the room is named again, from 0 to 1.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct CompositionChange(f32);

impl CompositionChange {
    pub fn new(value: f32) -> Result<Self, String> {
        if (0.0..=1.0).contains(&value) {
            Ok(Self(value))
        } else {
            Err(format!("{value} is not a share between 0 and 1"))
        }
    }

    pub fn get(self) -> f32 {
        self.0
    }
}

impl<'de> Deserialize<'de> for CompositionChange {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        CompositionChange::new(f32::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoomNamerConfig {
    pub model: ModelName,
    pub effort: Effort,
    pub system_prompt: SystemPrompt,
    /// A named room is named again once more than this share of its objects has changed.
    pub rename_after_change: CompositionChange,
}

impl RoomNamerConfig {
    pub fn profile(&self) -> ModelProfile {
        ModelProfile { model: self.model.clone(), effort: self.effort, system_prompt: self.system_prompt.clone() }
    }
}

/// When a model call counts as hung, and how hard to stop it.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WatchdogConfig {
    /// A call with no event from claude for this long is interrupted.
    #[serde(rename = "silent_turn_s", deserialize_with = "seconds")]
    pub silent_turn: Duration,
    /// After an interrupt with no result for this long, the process gets SIGTERM.
    #[serde(rename = "interrupt_grace_s", deserialize_with = "seconds")]
    pub interrupt_grace: Duration,
    /// After SIGTERM with the process still there for this long, it gets SIGKILL.
    #[serde(rename = "terminate_grace_s", deserialize_with = "seconds")]
    pub terminate_grace: Duration,
    /// How long a new process may take to show it is up.
    #[serde(rename = "startup_s", deserialize_with = "seconds")]
    pub startup: Duration,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoalDefaults {
    #[serde(deserialize_with = "metres")]
    pub standoff_m: NonNegative,
    #[serde(deserialize_with = "metres")]
    pub max_speed_m_s: NonNegative,
    #[serde(rename = "deadline_s", deserialize_with = "seconds")]
    pub deadline: Duration,
}

impl GoalDefaults {
    pub fn approach(&self) -> Approach {
        Approach { standoff_m: self.standoff_m, max_speed_m_s: self.max_speed_m_s, within: self.deadline }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct McpConfig {
    /// The port on 127.0.0.1 the MCP endpoint listens on; 0 takes any free one.
    pub port: u16,
    /// The most rows `query_sql` returns.
    pub sql_row_limit: NonZeroUsize,
}

/// Where the world store copies itself, and how often.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorldStoreBackup {
    pub path: PathBuf,
    #[serde(rename = "every_s", deserialize_with = "seconds")]
    pub every: Duration,
}

/// The world store the dora node answers from. The console answers from the in-memory house and does not open it.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorldStoreSettings {
    /// The SQLite database; created on first start. Only one process may have it open.
    pub path: PathBuf,
    /// Periodic copies of the database; none when the section is left out.
    pub backup: Option<WorldStoreBackup>,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpatialConfig {
    pub claude: ClaudeConfig,
    pub answering: AnsweringConfig,
    pub reconciler: ModelProfile,
    pub room_namer: RoomNamerConfig,
    pub watchdog: WatchdogConfig,
    pub goals: GoalDefaults,
    pub mcp: McpConfig,
    pub world_store: WorldStoreSettings,
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("reading {path}: {source}")]
    Read { path: PathBuf, source: std::io::Error },
    #[error("parsing the settings: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("exactly one answerer must be fast, not {0}")]
    FastAnswerers(usize),
}

impl SpatialConfig {
    pub fn parse(text: &str) -> Result<Self, ConfigError> {
        let config: SpatialConfig = toml::from_str(text)?;
        let fast = config.answering.answerers.iter().filter(|a| a.role == AnswererRole::Fast).count();
        if fast != 1 {
            return Err(ConfigError::FastAnswerers(fast));
        }
        Ok(config)
    }

    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Read { path: path.to_owned(), source })?;
        Self::parse(&text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example() -> String {
        std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("spatial-agent.example.toml")).unwrap()
    }

    #[test]
    fn the_example_settings_answer_with_haiku_first_then_opus_and_fable() {
        let config = SpatialConfig::parse(&example()).unwrap();
        let answerers: Vec<(&str, AnswererRole)> =
            config.answering.answerers.iter().map(|a| (a.model.0.as_str(), a.role)).collect();
        assert_eq!(
            answerers,
            [
                ("claude-haiku-5-5", AnswererRole::Fast),
                ("claude-opus-5-5", AnswererRole::Deliberate),
                ("claude-fable-5-2", AnswererRole::Deliberate),
            ]
        );
        assert_eq!((config.reconciler.model.0.as_str(), config.reconciler.effort), ("claude-haiku-5-5", Effort::Low));
        assert!(config.answering.system_prompt.0.contains("WORLD DIGEST"));
        assert_eq!(config.claude.mcp_tool_timeout, Duration::from_secs(5));
        assert_eq!(config.world_store.path, Path::new("/tmp/spatial-agent-world.sqlite"));
        assert_eq!(config.world_store.backup, None);
    }

    #[test]
    fn the_container_settings_parse() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("spatial-agent.container.toml");
        let config = SpatialConfig::load(&path).unwrap();
        assert_eq!(config.claude.work_dir, Path::new("/var/lib/spatial-agent/claude"));
        assert_eq!(config.world_store.path, Path::new("/var/lib/spatial-agent/world.sqlite"));
        let backup = config.world_store.backup.unwrap();
        assert_eq!((backup.path.as_path(), backup.every), (Path::new("/var/lib/spatial-agent/backup/world.sqlite"), Duration::from_secs(600)));
    }

    #[test]
    fn two_fast_answerers_are_an_error() {
        let text = example().replace("role = \"deliberate\"", "role = \"fast\"");
        assert!(matches!(SpatialConfig::parse(&text), Err(ConfigError::FastAnswerers(3))));
    }

    #[test]
    fn an_unknown_setting_is_an_error() {
        let text = example().replace("[reconciler]", "[reconciler]\ntemperature = 0.2");
        let error = SpatialConfig::parse(&text).unwrap_err();
        assert!(error.to_string().contains("temperature"), "{error}");
    }

    #[test]
    fn a_change_share_above_one_is_an_error() {
        let text = example().replace("rename_after_change = 0.5", "rename_after_change = 1.5");
        let error = SpatialConfig::parse(&text).unwrap_err();
        assert!(error.to_string().contains("not a share between 0 and 1"), "{error}");
    }
}
