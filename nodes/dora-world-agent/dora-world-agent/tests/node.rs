//! The node without dora: encoded map, object and pose messages go into a real world store, a space question is
//! answered by `fake-claude` from that store, and the goal the answer sends is recorded in it.

use std::future::Future;
use std::num::NonZeroUsize;
use std::path::Path;
use std::time::Duration;

use dora_node_api::arrow::array::StructArray;
use dora_world_agent::node::{Outputs, SpatialNode, SPACE_QUESTION_INPUT};
use dora_world_agent::world::{SqliteWorld, StoreInput};
use glam::{Vec2, Vec3};
use perception_wire::localization::{encode_keyframe_pose, FrameId, KeyframePose, Pose4, Tracking};
use perception_wire::navigation::{
    encode_goal_status, encode_space_question, Asker, GoalState, GoalStatus, MapVersion, NavGoal, NonNegative, ObjectId,
    PlaceId, QuestionId, Reference, Referent, RoomId, SpaceAnswer, SpaceQuestion, UnixNanos,
};
use perception_wire::objects::{encode_object_update, ObjectExtent, ObjectUpdate, ObjectUpdateKind};
use perception_wire::space_map::{
    encode_free_space, encode_topology, Doorway, DoorwayId, FreeSpace, HeightBand, Place, PlaceEdge, Topology, TopologyRoom,
};
use world_agent::clock;
use world_agent::config::{
    AnswererConfig, AnswererRole, AnsweringConfig, ClaudeConfig, CompositionChange, Effort, GoalDefaults, McpConfig, ModelName,
    ModelProfile, RoomNamerConfig, SpatialConfig, SystemPrompt, WatchdogConfig, WorldStoreSettings,
};
use world_agent::world::SqlValue;
use world_store::StoreConfig;

const FAST: &str = "fast-model";
const VERSION: MapVersion = MapVersion(7);
const KITCHEN: RoomId = RoomId(1);
const HALLWAY: RoomId = RoomId(2);
const TABLE: ObjectId = ObjectId(42);
const MUG: ObjectId = ObjectId(51);
const QUESTION: QuestionId = QuestionId(9);

fn model(name: &str) -> ModelName {
    ModelName(name.to_owned())
}

/// One fast answerer and nothing else, so its answer is the final one.
fn config(dir: &Path) -> SpatialConfig {
    SpatialConfig {
        claude: ClaudeConfig {
            executable: env!("CARGO_BIN_EXE_node-fake-claude").into(),
            work_dir: dir.join("claude"),
            mcp_tool_timeout: Duration::from_secs(5),
        },
        answering: AnsweringConfig {
            system_prompt: SystemPrompt("You answer questions.".to_owned()),
            deadline: Duration::from_secs(10),
            answerers: vec![AnswererConfig { model: model(FAST), effort: Effort::Low, role: AnswererRole::Fast }],
        },
        reconciler: ModelProfile {
            model: model("reconciler"),
            effort: Effort::Low,
            system_prompt: SystemPrompt("You reconcile. [fake:reply={\"decision\": \"keep\"}]".to_owned()),
        },
        room_namer: RoomNamerConfig {
            model: model("namer"),
            effort: Effort::Low,
            system_prompt: SystemPrompt("You name rooms. [fake:reply=kitchen]".to_owned()),
            rename_after_change: CompositionChange::new(0.5).unwrap(),
        },
        watchdog: WatchdogConfig {
            silent_turn: Duration::from_secs(5),
            interrupt_grace: Duration::from_millis(300),
            terminate_grace: Duration::from_millis(300),
            startup: Duration::from_secs(10),
        },
        goals: GoalDefaults {
            standoff_m: NonNegative::new("standoff_m", 0.5).unwrap(),
            max_speed_m_s: NonNegative::new("max_speed_m_s", 0.3).unwrap(),
            deadline: Duration::from_secs(60),
        },
        mcp: McpConfig { port: 0, sql_row_limit: NonZeroUsize::new(10).unwrap() },
        world_store: WorldStoreSettings { path: dir.join("world.sqlite"), backup: None },
    }
}

fn rect(x0: f32, y0: f32, x1: f32, y1: f32) -> Vec<Vec2> {
    vec![Vec2::new(x0, y0), Vec2::new(x1, y0), Vec2::new(x1, y1), Vec2::new(x0, y1)]
}

/// A kitchen and a hallway joined by one doorway, with a place in each and one at the doorway.
fn topology(built: UnixNanos) -> StructArray {
    let place = |id, room, x, y, clearance_m| Place { place_id: PlaceId(id), room_id: room, position: Vec2::new(x, y), clearance_m };
    let edge = |a, b, length_m| PlaceEdge { place_a: PlaceId(a), place_b: PlaceId(b), length_m, min_clearance_m: 0.8 };
    encode_topology(&Topology {
        map_version: VERSION,
        built,
        rooms: vec![
            TopologyRoom { room_id: KITCHEN, polygon: rect(0.0, 0.0, 4.0, 4.0), floor_m: 0.0, ceiling_m: Some(2.4) },
            TopologyRoom { room_id: HALLWAY, polygon: rect(4.0, 0.0, 6.0, 4.0), floor_m: 0.0, ceiling_m: Some(2.4) },
        ],
        doorways: vec![Doorway { doorway_id: DoorwayId(1), room_a: KITCHEN, room_b: HALLWAY, position: Vec2::new(4.0, 2.0), width_m: 0.9 }],
        places: vec![place(1, KITCHEN, 2.0, 3.0, 1.0), place(2, KITCHEN, 4.0, 2.0, 0.45), place(3, HALLWAY, 5.0, 2.0, 0.9)],
        edges: vec![edge(1, 2, 2.2), edge(2, 3, 1.0)],
    })
}

fn free_space(built: UnixNanos) -> StructArray {
    encode_free_space(&FreeSpace {
        map_version: VERSION,
        built,
        origin: Vec2::ZERO,
        resolution_m: 0.1,
        width: 60,
        height: 40,
        band: HeightBand { min_m: 0.05, max_m: 1.2 },
        clearance_mm: vec![1000; 60 * 40],
    })
}

fn appeared(id: ObjectId, label: &str, centre: Vec2, (long_m, short_m, height_m): (f32, f32, f32), at: UnixNanos) -> StructArray {
    let (half_long, half_short) = (long_m / 2.0, short_m / 2.0);
    let corners = rect(centre.x - half_long, centre.y - half_short, centre.x + half_long, centre.y + half_short);
    encode_object_update(&ObjectUpdate {
        kind: ObjectUpdateKind::Appeared,
        object_id: id,
        merged_into: None,
        label: label.to_owned(),
        score: 0.9,
        confidence: 0.8,
        seen_count: 5,
        frame_id: FrameId(3),
        captured: at,
        map_version: VERSION,
        extent: ObjectExtent {
            centre_m: Vec3::new(centre.x, centre.y, height_m / 2.0),
            footprint_m: [corners[0], corners[1], corners[2], corners[3]],
            long_m,
            short_m,
            height_m,
            top_m: height_m,
        },
        crop_jpeg: None,
    })
}

fn keyframe_pose(at: UnixNanos) -> StructArray {
    encode_keyframe_pose(&KeyframePose {
        frame_id: FrameId(4),
        captured: at,
        map_version: VERSION,
        pose: Pose4 { x_m: 3.2, y_m: 2.5, z_m: 0.0, yaw_rad: 1.5 },
        tracking: Tracking::Tracking,
        residual_m: 0.01,
        inlier_fraction: 0.9,
    })
}

/// Polls until `check` gives a value, for what the store's writer commits in its own time.
async fn eventually<T, F: Future<Output = Option<T>>>(what: &str, mut check: impl FnMut() -> F) -> T {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(value) = check().await {
            return value;
        }
        assert!(tokio::time::Instant::now() < deadline, "{what} did not happen");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

async fn rows(world: &SqliteWorld, sql: &str) -> Vec<Vec<SqlValue>> {
    world.query_text(sql, NonZeroUsize::new(10).unwrap()).await.unwrap().rows
}

/// Every answer up to the final one, recording each goal sent on the way as the dora loop does.
async fn thread(spatial: &SpatialNode<SqliteWorld>, outputs: &mut Outputs) -> (Vec<SpaceAnswer>, Vec<NavGoal>) {
    let (mut answers, mut goals) = (Vec::new(), Vec::new());
    tokio::time::timeout(Duration::from_secs(20), async {
        loop {
            tokio::select! {
                Some(sent) = outputs.goals.recv() => goals.push(spatial.on_goal(sent).unwrap()),
                Some(answer) = outputs.answers.recv() => {
                    let is_final = answer.is_final;
                    answers.push(answer);
                    if is_final {
                        return;
                    }
                }
            }
        }
    })
    .await
    .unwrap_or_else(|_| panic!("no final answer; so far {answers:#?}"));
    while let Ok(sent) = outputs.goals.try_recv() {
        goals.push(spatial.on_goal(sent).unwrap());
    }
    (answers, goals)
}

#[tokio::test(flavor = "multi_thread")]
async fn a_question_is_answered_from_the_ingested_world_and_its_goal_is_recorded() {
    let dir = tempfile::tempdir().unwrap();
    let config = config(dir.path());
    let mut store_config = StoreConfig::at(&config.world_store.path);
    store_config.batch_interval = Duration::from_millis(20);
    let world = SqliteWorld::open_with(store_config).unwrap();
    let (spatial, mut outputs) = SpatialNode::start(&config, world.clone()).await.unwrap();

    let now = clock::now();
    spatial.on_input(StoreInput::Topology.input_id(), &topology(now)).await.unwrap();
    spatial.on_input(StoreInput::FreeSpace.input_id(), &free_space(now)).await.unwrap();
    spatial.on_input(StoreInput::ObjectUpdate.input_id(), &appeared(TABLE, "table", Vec2::new(2.0, 1.5), (1.6, 0.9, 0.74), now)).await.unwrap();
    spatial.on_input(StoreInput::ObjectUpdate.input_id(), &appeared(MUG, "mug", Vec2::new(2.2, 1.6), (0.1, 0.1, 0.1), now)).await.unwrap();
    spatial.on_input(StoreInput::KeyframePose.input_id(), &keyframe_pose(now)).await.unwrap();
    // The map is rebuilt with the objects in it, so the kitchen has something to be named by.
    spatial.on_input(StoreInput::Topology.input_id(), &topology(now)).await.unwrap();
    let named = eventually("the kitchen being named", || async {
        let named = rows(&world, "SELECT room_id, name FROM room_name").await;
        (!named.is_empty()).then_some(named)
    })
    .await;
    assert_eq!(named, [vec![SqlValue::Integer(1), SqlValue::Text("kitchen".to_owned())]]);

    let question = SpaceQuestion {
        question_id: QUESTION,
        asked: now,
        asker: Asker("planner".to_owned()),
        text: [
            "Take me to my mug.",
            &format!(r#"fake {FAST} call send_goal {{"kind": "object", "target": "o42"}}"#),
            &format!("fake {FAST} say The mug o51 is on the table o42 in r1; going to the table."),
        ]
        .join("\n"),
    };
    spatial.on_input(SPACE_QUESTION_INPUT, &encode_space_question(&question)).await.unwrap();
    let (answers, goals) = thread(&spatial, &mut outputs).await;

    assert_eq!(answers.len(), 1, "{answers:#?}");
    let answer = &answers[0];
    assert!(answer.is_final);
    assert_eq!(answer.text, "The mug o51 is on the table o42 in r1; going to the table.");
    assert_eq!(
        answer.references,
        [
            Reference { referent: Referent::Object(MUG), x_m: 2.2, y_m: 1.6 },
            Reference { referent: Referent::Object(TABLE), x_m: 2.0, y_m: 1.5 },
            Reference { referent: Referent::Room(KITCHEN), x_m: 2.0, y_m: 2.0 },
        ]
    );
    let [goal] = goals.as_slice() else { panic!("one goal, not {goals:#?}") };
    assert_eq!(answer.goal_id, Some(goal.goal_id));

    let status = GoalStatus { goal_id: goal.goal_id, state: GoalState::Active, updated: clock::now() };
    spatial.on_input(StoreInput::GoalStatus.input_id(), &encode_goal_status(&status)).await.unwrap();
    let recorded = eventually("the goal's status being recorded", || async {
        let recorded = rows(&world, "SELECT goal_id, agent, model, kind, target_id, state FROM goal").await;
        recorded.first().is_some_and(|row| row[5] == SqlValue::Text("active".to_owned())).then_some(recorded)
    })
    .await;
    assert_eq!(
        recorded,
        [vec![
            SqlValue::Integer(goal.goal_id.0 as i64),
            SqlValue::Text("answerer 0".to_owned()),
            SqlValue::Text(FAST.to_owned()),
            SqlValue::Text("object".to_owned()),
            SqlValue::Integer(TABLE.0 as i64),
            SqlValue::Text("active".to_owned()),
        ]]
    );
    spatial.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn an_input_that_does_not_decode_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let config = config(dir.path());
    let world = SqliteWorld::open(&config.world_store).unwrap();
    let (spatial, _outputs) = SpatialNode::start(&config, world).await.unwrap();
    let error = spatial.on_input(StoreInput::Topology.input_id(), &free_space(clock::now())).await.unwrap_err();
    assert!(format!("{error:#}").contains("decoding a topology"), "{error:#}");
    let error = spatial.on_input("keep_out", &free_space(clock::now())).await.unwrap_err();
    assert!(error.to_string().contains("unexpected input keep_out"), "{error}");
    spatial.shutdown().await.unwrap();
}
