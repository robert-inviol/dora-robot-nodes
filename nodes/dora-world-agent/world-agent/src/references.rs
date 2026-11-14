//! What an answer refers to: the ids it names (objects, places, people tracks and rooms), in the order it names them,
//! and its goal's target, each placed in the map from the snapshot the question was answered from. A room is placed
//! at its centroid.

use perception_wire::navigation::{GoalTarget, LookTarget, ObjectId, PlaceId, Reference, Referent, RoomId, TrackId};

use crate::world::{EntityRef, Point, WorldSnapshot};

fn object_position(snapshot: &WorldSnapshot, id: ObjectId) -> Option<Point> {
    snapshot.rooms.iter().flat_map(|r| &r.objects).find(|o| o.id == id).map(|o| o.position)
}

fn place_position(snapshot: &WorldSnapshot, id: PlaceId) -> Option<Point> {
    snapshot.rooms.iter().flat_map(|r| &r.places).find(|p| p.id == id).map(|p| p.position)
}

fn track_position(snapshot: &WorldSnapshot, id: TrackId) -> Option<Point> {
    snapshot.people.iter().find(|t| t.id == id).map(|t| t.position)
}

fn room_position(snapshot: &WorldSnapshot, id: RoomId) -> Option<Point> {
    snapshot.rooms.iter().find(|r| r.id == Some(id)).map(|r| r.centroid)
}

/// A room id as the digest writes it: `r` and a number.
fn parse_room_id(word: &str) -> Option<RoomId> {
    let digits = word.strip_prefix('r')?;
    match !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()) {
        true => digits.parse().ok().map(RoomId),
        false => None,
    }
}

/// The referent a word of an answer names, if it is an id other than the robot.
fn named_referent(word: &str) -> Option<Referent> {
    if let Some(room) = parse_room_id(word) {
        return Some(Referent::Room(room));
    }
    match word.parse::<EntityRef>().ok()? {
        EntityRef::Object(id) => Some(Referent::Object(id)),
        EntityRef::Place(id) => Some(Referent::Place(id)),
        EntityRef::Track(id) => Some(Referent::Track(id)),
        EntityRef::Robot => None,
    }
}

fn locate(snapshot: &WorldSnapshot, referent: Referent) -> Option<Point> {
    match referent {
        Referent::Object(id) => object_position(snapshot, id),
        Referent::Place(id) => place_position(snapshot, id),
        Referent::Track(id) => track_position(snapshot, id),
        Referent::Room(id) => room_position(snapshot, id),
        Referent::Pose => None,
    }
}

fn goal_reference(target: GoalTarget) -> Option<(Referent, Option<Point>)> {
    match target {
        GoalTarget::Object(id) | GoalTarget::LookAt(LookTarget::Object(id)) => Some((Referent::Object(id), None)),
        GoalTarget::Place(id) => Some((Referent::Place(id), None)),
        GoalTarget::FollowTrack(id) => Some((Referent::Track(id), None)),
        GoalTarget::Pose(pose) | GoalTarget::LookAt(LookTarget::Point(pose)) => {
            Some((Referent::Pose, Some(Point { x_m: pose.x_m, y_m: pose.y_m })))
        }
        GoalTarget::Stop => None,
    }
}

/// The ids the text names that the snapshot knows, then the goal's target if the text does not name it. An id the
/// snapshot does not know is left out and logged: the model invented it.
pub fn references(text: &str, goal: Option<GoalTarget>, snapshot: &WorldSnapshot) -> Vec<Reference> {
    let named = text.split(|c: char| !c.is_ascii_alphanumeric()).filter_map(named_referent).map(|referent| (referent, None));
    let mut found: Vec<Reference> = Vec::new();
    for (referent, given) in named.chain(goal.and_then(goal_reference)) {
        if referent != Referent::Pose && found.iter().any(|r| r.referent == referent) {
            continue;
        }
        match given.or_else(|| locate(snapshot, referent)) {
            Some(point) => found.push(Reference { referent, x_m: point.x_m, y_m: point.y_m }),
            None => tracing::warn!(?referent, "an answer names an id the world model does not know"),
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use perception_wire::navigation::{MapPose, UnixNanos};

    use super::*;
    use crate::fake_world::FakeWorld;
    use crate::world::WorldModel;

    async fn snapshot() -> WorldSnapshot {
        FakeWorld::house_at(UnixNanos(1_794_610_000_000_000_000)).snapshot().await.unwrap()
    }

    #[tokio::test]
    async fn named_ids_are_placed_in_order_once_each() {
        let refs = references("The mug o51 is on o42, near p1; o51 again. Ask t7.", None, &snapshot().await);
        assert_eq!(
            refs,
            vec![
                Reference { referent: Referent::Object(ObjectId(51)), x_m: 2.2, y_m: 1.6 },
                Reference { referent: Referent::Object(ObjectId(42)), x_m: 2.0, y_m: 1.5 },
                Reference { referent: Referent::Place(PlaceId(1)), x_m: 2.0, y_m: 2.5 },
                Reference { referent: Referent::Track(TrackId(7)), x_m: 8.1, y_m: 2.0 },
            ]
        );
    }

    #[tokio::test]
    async fn a_named_room_is_placed_at_its_centroid() {
        let refs = references("It is in r3, not r9 or rx or r.", None, &snapshot().await);
        assert_eq!(refs, vec![Reference { referent: Referent::Room(RoomId(3)), x_m: 9.0, y_m: 3.0 }]);
    }

    #[tokio::test]
    async fn invented_ids_are_left_out_and_the_goal_target_is_added() {
        let refs = references("Try o999 or robot.", Some(GoalTarget::Place(PlaceId(5))), &snapshot().await);
        assert_eq!(refs.iter().map(|r| r.referent).collect::<Vec<_>>(), vec![Referent::Place(PlaceId(5))]);
        let pose = GoalTarget::Pose(MapPose { x_m: 1.5, y_m: -0.5, yaw_rad: 0.0 });
        assert_eq!(references("Over there.", Some(pose), &snapshot().await), vec![Reference { referent: Referent::Pose, x_m: 1.5, y_m: -0.5 }]);
    }
}
