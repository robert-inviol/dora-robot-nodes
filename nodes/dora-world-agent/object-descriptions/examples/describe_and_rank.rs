//! Runs the real describe and embed path on one view and ranks queries against it, as `find` would score them:
//!
//! ```
//! GEMINI_API_KEY=... cargo run -p object-descriptions --example describe_and_rank -- \
//!     --settings world-agent/spatial-agent.toml --image cup.jpg --size 12x8x10 --labels cup,mug "red cup" "blue mug"
//! ```
//!
//! The settings file must have an `[object_descriptions]` section; its `[claude]` executable and credentials are
//! used, with a fresh work directory. It prints the description, the length of each vector, and for each query its
//! similarity to the description and to the view and the score `find` gives it.

use std::sync::Arc;

use object_descriptions::describer::{describer_profile, ClaudeDescriber, Describer};
use object_descriptions::embedder::{Embedder, EmbeddingInput};
use object_descriptions::facts::ObjectFacts;
use object_descriptions::gemini::{GeminiApiKey, GeminiEmbedder, GEMINI_API_KEY_VARIABLE};
use object_descriptions::try_cli::{TryArguments, USAGE};
use perception_wire::navigation::ObjectId;
use world_agent::claude_code::ClaudeCodeClient;
use world_agent::config::SpatialConfig;
use world_agent::identity::AgentRegistry;
use world_agent::process::Launcher;
use world_model::{MatchScores, WordShare};
use world_store::JpegImage;

type Failure = Box<dyn std::error::Error + Send + Sync>;

#[tokio::main]
async fn main() -> Result<(), Failure> {
    tracing_subscriber::fmt().with_env_filter("warn").with_writer(std::io::stderr).init();
    let arguments = TryArguments::parse(std::env::args().skip(1)).map_err(|error| format!("{error}\n{USAGE}"))?;
    let config = SpatialConfig::load(&arguments.settings)?;
    let descriptions = config.object_descriptions.as_ref().ok_or("the settings file has no [object_descriptions] section")?;
    let key = GeminiApiKey::from_variable(std::env::var_os(GEMINI_API_KEY_VARIABLE)).ok_or(format!("{GEMINI_API_KEY_VARIABLE} is not set"))?;
    let view = JpegImage(std::fs::read(&arguments.image)?);
    if !view.0.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return Err(format!("{} is not a JPEG", arguments.image.display()).into());
    }

    let mut claude = config.claude.clone();
    claude.work_dir = std::env::temp_dir().join(format!("describe-and-rank-{}", std::process::id()));
    std::fs::create_dir_all(&claude.work_dir)?;
    let registry = Arc::new(AgentRegistry::default());
    let launcher = Launcher::new(claude, describer_profile(descriptions), registry.clone(), config.watchdog.startup);
    let client = Arc::new(ClaudeCodeClient::start(launcher, registry, config.watchdog));
    let describer = ClaudeDescriber::new(client.clone(), descriptions.describe_timeout);
    let embedder = GeminiEmbedder::new(key, &descriptions.embedding)?;

    let facts = ObjectFacts { size: arguments.size, labels: arguments.labels.clone() };
    println!("facts: {}", facts.line());
    let caption = describer.describe(ObjectId(0), &view, &facts).await?;
    client.stop();
    println!("description: {}", serde_json::to_string_pretty(&caption)?);
    let description = embedder.embed(EmbeddingInput::Description(&caption)).await?;
    let image = embedder.embed(EmbeddingInput::View(&view)).await?;
    println!("description vector: {} values; view vector: {} values ({})", description.dimensions(), image.dimensions(), embedder.model());

    let mut texts: Vec<&str> = arguments.labels.iter().map(String::as_str).collect();
    texts.extend([&caption.name, &caption.category, &caption.description, &caption.colour, &caption.material].map(String::as_str));
    println!("{:<30} {:>11} {:>6} {:>6} {:>6}", "query", "description", "view", "words", "score");
    for query in &arguments.queries {
        let vector = embedder.embed(EmbeddingInput::Query(query)).await?;
        let matched = MatchScores {
            words: WordShare::of_query(query, &texts),
            description: Some(vector.similarity(&description)),
            view: Some(vector.similarity(&image)),
        };
        println!(
            "{:<30} {:>11.3} {:>6.3} {:>6.2} {:>6.3}",
            query,
            vector.similarity(&description).get(),
            vector.similarity(&image).get(),
            matched.words.get(),
            matched.score_by_meaning()
        );
    }
    Ok(())
}
