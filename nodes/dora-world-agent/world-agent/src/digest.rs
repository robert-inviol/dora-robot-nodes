//! The world digest at the start of every user message: one line per entity, grouped by room (each room headed by
//! its id, such as `r3`, unless its objects stand where no room is mapped), ids stable across
//! questions, ages relative to `as_of`. It is built once when a question arrives, from one snapshot, and every model
//! that answers or reconciles the question gets the same text at the top of its user message, never in the system
//! prompt.

use std::fmt::{self, Write};

use perception_wire::navigation::{GoalTarget, LookTarget, UnixNanos};

use crate::clock;
use crate::goals::GoalRecord;
use crate::world::{PersonTrack, Point, TrackLiveness, WorldSnapshot};

pub const DIGEST_HEADING: &str = "WORLD DIGEST";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Digest(String);

impl Digest {
    pub fn as_str(&self) -> &str {
        &self.0
    }

    #[cfg(test)]
    pub(crate) fn from_text(text: &str) -> Self {
        Self(text.to_owned())
    }
}

impl fmt::Display for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// The text of an answerer's user message: the digest, a blank line, then the question.
pub fn compose_question(digest: &Digest, question: &str) -> String {
    format!("{}\n\n{}", digest.0, question)
}

fn age(seconds: f32) -> String {
    match seconds {
        s if s < 90.0 => format!("{s:.0}s"),
        s if s < 90.0 * 60.0 => format!("{:.0}m", s / 60.0),
        s if s < 48.0 * 3600.0 => format!("{:.0}h", s / 3600.0),
        s => format!("{:.0}d", s / 86_400.0),
    }
}

fn at(p: Point) -> String {
    format!("({:.2},{:.2})", p.x_m, p.y_m)
}

fn person(track: &PersonTrack, as_of: UnixNanos) -> String {
    let seen = age(clock::seconds_between(track.last_seen, as_of));
    match track.liveness {
        TrackLiveness::Live => format!(
            "{} {} {} heading {:.0}deg {:.1}m/s seen {seen} ago",
            track.id, track.room, at(track.position), track.heading_deg, track.speed_m_s
        ),
        TrackLiveness::Lost => format!(
            "{} lost {seen} ago, last {} {} heading {:.0}deg",
            track.id, track.room, at(track.position), track.heading_deg
        ),
    }
}

/// A goal's target as the digest and the reconciler's prompt name it.
pub fn describe_target(target: GoalTarget) -> String {
    match target {
        GoalTarget::Pose(p) => format!("pose ({:.2},{:.2})", p.x_m, p.y_m),
        GoalTarget::Object(id) => format!("object {id}"),
        GoalTarget::Place(id) => format!("place {id}"),
        GoalTarget::FollowTrack(id) => format!("follow {id}"),
        GoalTarget::LookAt(LookTarget::Object(id)) => format!("look at {id}"),
        GoalTarget::LookAt(LookTarget::Point(p)) => format!("look at ({:.2},{:.2})", p.x_m, p.y_m),
        GoalTarget::Stop => "stop".to_owned(),
    }
}

fn goal_line(record: &GoalRecord, as_of: UnixNanos) -> String {
    let target = describe_target(record.goal.target);
    let state = record.status.map_or("sent, not yet acknowledged".to_owned(), |s| s.state.to_string());
    format!(
        "{} {target} {state} (sent {} ago for {})",
        record.goal.goal_id,
        age(clock::seconds_between(record.issued, as_of)),
        record.question
    )
}

pub fn render(snapshot: &WorldSnapshot, running_goal: Option<&GoalRecord>) -> Digest {
    let as_of = snapshot.stamp.as_of;
    let mut text = String::new();
    let mut line = |content: String| {
        text.push_str(&content);
        text.push('\n');
    };
    line(format!(
        "{DIGEST_HEADING} as_of {} map {} (fused {} ago)",
        clock::iso_time(as_of),
        snapshot.stamp.map_version,
        age(clock::seconds_between(snapshot.map_fused, as_of))
    ));
    let robot = &snapshot.robot;
    line(format!(
        "robot: {} ({:.2},{:.2}) yaw {:.0}deg",
        robot.room,
        robot.pose.x_m,
        robot.pose.y_m,
        robot.pose.yaw_rad.to_degrees()
    ));
    line(format!("goal: {}", running_goal.map_or("none running".to_owned(), |g| goal_line(g, as_of))));
    let people: Vec<String> = snapshot.people.iter().map(|t| person(t, as_of)).collect();
    line(format!("people: {}", if people.is_empty() { "none seen".to_owned() } else { people.join(" | ") }));
    for stale in &snapshot.stale_sources {
        line(format!("STALE: {} silent {:.0}s; do not rely on it", stale.name, stale.silent_s));
    }
    for room in &snapshot.rooms {
        let mut heading = match room.id {
            Some(id) => format!("{id} {}", room.name),
            None => room.name.to_string(),
        };
        if !room.doorways.is_empty() {
            let doors: Vec<String> = room.doorways.iter().map(|d| format!("{}->{}", d.place, d.to)).collect();
            write!(heading, " [doors: {}]", doors.join(", ")).expect("writing to a string");
        }
        if !room.places.is_empty() {
            let places: Vec<String> = room.places.iter().map(|p| format!("{} {}", p.id, p.name)).collect();
            write!(heading, " [places: {}]", places.join(", ")).expect("writing to a string");
        }
        line(heading);
        for object in &room.objects {
            line(format!(
                "  {} {} {:.1}x{:.1} h{:.2} {} seen {} ago",
                object.id,
                object.label,
                object.size.long_m,
                object.size.short_m,
                object.size.height_m,
                at(object.position),
                age(clock::seconds_between(object.last_seen, as_of))
            ));
        }
    }
    let changes: Vec<String> = snapshot
        .recent_changes
        .iter()
        .map(|c| {
            let what = match &c.kind {
                crate::world::ChangeKind::Appeared => format!("appeared in {}", c.room),
                crate::world::ChangeKind::Moved { from, to } => format!("moved {from}->{to}"),
                crate::world::ChangeKind::Disappeared => format!("gone from {}", c.room),
            };
            format!("{} ago {} {} {what}", age(clock::seconds_between(c.detected, as_of)), c.object, c.label)
        })
        .collect();
    line(format!("changes: {}", if changes.is_empty() { "none".to_owned() } else { changes.join(" ; ") }));
    text.pop();
    Digest(text)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use perception_wire::navigation::{
        GoalId, GoalReason, GoalState, GoalStatus, MapVersion, NavGoal, NonNegative, ObjectId, QuestionId,
    };

    use super::*;
    use crate::fake_world::FakeWorld;
    use crate::world::{StaleSource, WorldModel};

    const NOW: UnixNanos = UnixNanos(1_794_610_000_000_000_000);

    #[tokio::test]
    async fn the_digest_lists_the_robot_people_rooms_objects_and_changes() {
        let snapshot = FakeWorld::house_at(NOW).snapshot().await.unwrap();
        let digest = render(&snapshot, None);
        let expected = "\
WORLD DIGEST as_of 2026-11-13T22:46:40.0Z map v412 (fused 17s ago)
robot: kitchen (3.20,2.50) yaw 90deg
goal: none running
people: t7 living_room (8.10,2.00) heading 180deg 0.6m/s seen 0s ago | t3 lost 2m ago, last hallway (5.00,4.20) heading 90deg
r1 kitchen [doors: p2->hallway] [places: p1 kitchen middle]
  o42 table 1.6x0.9 h0.74 (2.00,1.50) seen 40s ago
  o43 chair 0.5x0.5 h0.90 (1.20,0.60) seen 40s ago
  o44 chair 0.5x0.5 h0.90 (2.80,0.60) seen 2m ago
  o51 mug 0.1x0.1 h0.10 (2.20,1.60) seen 5m ago
r2 hallway [doors: p2->kitchen, p4->living_room, p6->bedroom] [places: p3 hallway middle]
  o21 laundry basket 0.6x0.4 h0.50 (4.60,5.00) seen 25m ago
r3 living_room [doors: p4->hallway] [places: p5 living room middle]
  o12 sofa 2.1x0.9 h0.85 (9.00,5.20) seen 3m ago
  o13 armchair 0.9x0.8 h0.90 (7.40,4.90) seen 3m ago
r4 bedroom [doors: p6->hallway] [places: p7 bedroom middle]
  o20 bed 2.0x1.6 h0.50 (6.00,9.00) seen 60m ago
changes: 5m ago o51 mug appeared in kitchen ; 25m ago o21 laundry basket moved bedroom->hallway";
        assert_eq!(digest.as_str(), expected);
    }

    #[tokio::test]
    async fn a_running_goal_and_a_stale_source_are_called_out() {
        let mut snapshot = FakeWorld::house_at(NOW).snapshot().await.unwrap();
        snapshot.stale_sources.push(StaleSource { name: "people tracking".to_owned(), silent_s: 12.0 });
        let record = GoalRecord {
            goal: NavGoal {
                goal_id: GoalId(17),
                target: GoalTarget::Object(ObjectId(42)),
                standoff_m: NonNegative::new("standoff_m", 0.5).unwrap(),
                max_speed_m_s: NonNegative::new("max_speed_m_s", 0.3).unwrap(),
                deadline: clock::after(NOW, Duration::from_secs(60)),
                map_version: MapVersion(412),
            },
            question: QuestionId(31),
            issued: UnixNanos(NOW.0 - 12_000_000_000),
            status: Some(GoalStatus { goal_id: GoalId(17), state: GoalState::Blocked(GoalReason::PersonInPath), updated: NOW }),
        };
        let digest = render(&snapshot, Some(&record));
        assert!(digest.as_str().contains("\ngoal: g17 object o42 blocked (person_in_path) (sent 12s ago for q31)\n"));
        assert!(digest.as_str().contains("\nSTALE: people tracking silent 12s; do not rely on it\n"));
    }

    #[test]
    fn a_question_comes_after_the_digest() {
        let digest = Digest(format!("{DIGEST_HEADING} as_of now\nrobot: kitchen"));
        let text = compose_question(&digest, "Go to the table.");
        assert_eq!(text, "WORLD DIGEST as_of now\nrobot: kitchen\n\nGo to the table.");
    }
}
