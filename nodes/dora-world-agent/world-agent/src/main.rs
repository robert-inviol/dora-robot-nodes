//! The spatial agent against the in-memory house, without dora, for trying it by hand.
//!
//! `world-agent <settings.toml>` reads one question per line from standard input and prints every answer and nav goal
//! as it goes out. `/goal g17 succeeded` (or `active`, `blocked:person_in_path`, `failed:no_path`, ...) plays the
//! robot's report on a goal.

use std::path::PathBuf;
use std::sync::Arc;

use perception_wire::navigation::{Asker, GoalId, GoalReason, GoalState, GoalStatus, QuestionId, SpaceQuestion};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::mpsc;
use world_agent::clock;
use world_agent::config::SpatialConfig;
use world_agent::fake_world::FakeWorld;
use world_agent::goals::ChannelGoalSink;
use world_agent::service::SpatialService;

fn parse_goal_status(text: &str) -> Result<GoalStatus, String> {
    let usage = || format!("{text:?}: expected /goal g<id> <state>[:<reason>]");
    let (id, state) = text.split_once(' ').ok_or_else(usage)?;
    let goal_id = GoalId(id.strip_prefix('g').and_then(|n| n.parse().ok()).ok_or_else(usage)?);
    let (name, reason) = match state.trim().split_once(':') {
        Some((name, reason)) => (name, Some(reason.parse::<GoalReason>().map_err(|e| e.to_string())?)),
        None => (state.trim(), None),
    };
    let state = match (name, reason) {
        ("accepted", None) => GoalState::Accepted,
        ("active", None) => GoalState::Active,
        ("succeeded", None) => GoalState::Succeeded,
        ("blocked", Some(reason)) => GoalState::Blocked(reason),
        ("failed", Some(reason)) => GoalState::Failed(reason),
        _ => return Err(usage()),
    };
    Ok(GoalStatus { goal_id, state, updated: clock::now() })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info,rmcp=warn".into()))
        .with_writer(std::io::stderr)
        .init();
    let path: PathBuf = std::env::args_os().nth(1).ok_or("usage: world-agent <settings.toml>")?.into();
    let config = SpatialConfig::load(&path)?;
    let world = Arc::new(FakeWorld::house());
    let (goal_sender, mut goals) = mpsc::unbounded_channel();
    let (answer_sender, mut answers) = mpsc::unbounded_channel();
    let service = SpatialService::start(&config, world.clone(), ChannelGoalSink(goal_sender), FakeWorld::house(), answer_sender).await?;
    tokio::spawn(async move {
        loop {
            tokio::select! {
                Some(goal) = goals.recv() => println!("nav_goal {goal:?}"),
                Some(answer) = answers.recv() => println!(
                    "{} r{}{} {}: {} (goal {:?}, references {:?})",
                    answer.question_id,
                    answer.revision.0,
                    if answer.is_final { " final" } else { "" },
                    answer.model,
                    answer.text,
                    answer.goal_id,
                    answer.references,
                ),
                else => return,
            }
        }
    });
    eprintln!("ready: type a question, or /goal g<id> <state>");
    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    let mut next_question = 1;
    while let Some(line) = lines.next_line().await? {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(report) = line.strip_prefix("/goal ") {
            match parse_goal_status(report).map(|status| service.goal_status(status)) {
                Ok(Ok(())) => println!("recorded"),
                Ok(Err(error)) => println!("{error}"),
                Err(usage) => println!("{usage}"),
            }
            continue;
        }
        service.ask(SpaceQuestion {
            question_id: QuestionId(next_question),
            asked: clock::now(),
            asker: Asker("console".to_owned()),
            text: line.to_owned(),
        });
        next_question += 1;
    }
    service.shutdown().await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn goal_reports_parse_with_their_reasons() {
        assert_eq!(parse_goal_status("g17 succeeded").unwrap().state, GoalState::Succeeded);
        assert_eq!(parse_goal_status("g17 failed:no_path").unwrap().state, GoalState::Failed(GoalReason::NoPath));
        assert!(parse_goal_status("g17 failed").is_err());
        assert!(parse_goal_status("17 succeeded").is_err());
    }
}
