//! The page, the three.js build it runs, and its WebSocket. A page that connects gets the scene so far, then every
//! update as it happens. The keyframe clouds go only to a page that has asked for them with `{"clouds": true}`; it
//! gets the clouds so far when it asks, and `{"clouds": false}` stops them. The clouds so far go out between the
//! scene's updates, which always go first, so turning the clouds on never holds up the robot and the people; cloud
//! updates made since follow once the clouds so far are sent. A page that falls too far behind, stops reading, or
//! sends anything else, is disconnected with a reason, and reconnects for a fresh snapshot. Only the page's own
//! origin may open the WebSocket.

use std::collections::VecDeque;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::extract::ws::{CloseFrame, Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::http::header::{CONTENT_TYPE, HOST, ORIGIN};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use serde::Deserialize;
use tokio::sync::broadcast;

use crate::clouds::CloudStore;
use crate::scene::Scene;
use crate::view::{Layer, Outgoing, Payload};

const PAGE: &str = include_str!("page.html");
const THREE_MODULE: &str = include_str!("../vendor/three-0.160.0/three.module.js");
const ORBIT_CONTROLS: &str = include_str!("../vendor/three-0.160.0/OrbitControls.js");

/// Updates a page may fall behind by before it is disconnected.
const UPDATE_BACKLOG: usize = 4096;

/// How long one message may take to reach a page before the page is taken to have stopped reading.
pub const SEND_DEADLINE: Duration = Duration::from_secs(5);

/// WebSocket close code for a policy-driven close.
const CLOSE_POLICY: u16 = 1008;

/// The scene and the keyframe clouds, each with the channel its updates go out on. Each is changed and its updates
/// sent under its own lock, so a page that takes a snapshot and subscribes under the same lock misses none and sees
/// none twice; the two locks are separate so that carving the clouds never holds up the robot and the people.
#[derive(Clone)]
pub struct SharedView {
    scene: Arc<Mutex<Scene>>,
    clouds: Arc<Mutex<CloudStore>>,
    updates: broadcast::Sender<Outgoing>,
    cloud_updates: broadcast::Sender<Outgoing>,
}

impl SharedView {
    pub fn new(scene: Scene, clouds: CloudStore) -> SharedView {
        SharedView {
            scene: Arc::new(Mutex::new(scene)),
            clouds: Arc::new(Mutex::new(clouds)),
            updates: broadcast::channel(UPDATE_BACKLOG).0,
            cloud_updates: broadcast::channel(UPDATE_BACKLOG).0,
        }
    }

    /// Changes the scene and sends the updates the change made to every connected page. The change should be
    /// quick: heavy work belongs before it.
    pub fn apply_scene(&self, change: impl FnOnce(&mut Scene) -> Vec<Outgoing>) {
        let mut scene = self.scene.lock().expect("a scene change panicked");
        self.send_all(change(&mut scene));
    }

    /// Changes the keyframe clouds and sends the updates the change made to every page that shows them.
    pub fn apply_clouds(&self, change: impl FnOnce(&mut CloudStore) -> Vec<Outgoing>) {
        let mut clouds = self.clouds.lock().expect("a cloud change panicked");
        self.send_all(change(&mut clouds));
    }

    fn send_all(&self, updates: Vec<Outgoing>) {
        for update in updates {
            let channel = match update.layer {
                Layer::Always => &self.updates,
                Layer::Clouds => &self.cloud_updates,
            };
            // No page connected is not an error: the update is in the scene for the next one.
            let _ = channel.send(update);
        }
    }

    /// Reads the scene as it is now.
    pub fn read_scene<T>(&self, read: impl FnOnce(&Scene) -> T) -> T {
        read(&self.scene.lock().expect("a scene change panicked"))
    }

    fn connect(&self) -> (Vec<Outgoing>, broadcast::Receiver<Outgoing>) {
        let scene = self.scene.lock().expect("a scene change panicked");
        (scene.snapshot(), self.updates.subscribe())
    }

    fn subscribe_clouds(&self) -> CloudFeed {
        let clouds = self.clouds.lock().expect("a cloud change panicked");
        CloudFeed { so_far: clouds.snapshot().into(), updates: self.cloud_updates.subscribe() }
    }
}

/// The keyframe clouds for a page that shows them: the clouds so far still to send, then every cloud update.
struct CloudFeed {
    so_far: VecDeque<Outgoing>,
    updates: broadcast::Receiver<Outgoing>,
}

impl CloudFeed {
    async fn next(&mut self) -> Result<Outgoing, broadcast::error::RecvError> {
        match self.so_far.pop_front() {
            Some(outgoing) => Ok(outgoing),
            None => self.updates.recv().await,
        }
    }
}

/// The next cloud message for a page, or never while it does not show the clouds.
async fn next_cloud(feed: &mut Option<CloudFeed>) -> Result<Outgoing, broadcast::error::RecvError> {
    match feed {
        Some(feed) => feed.next().await,
        None => std::future::pending().await,
    }
}

/// Closes a page that fell behind, with the reason.
async fn on_lagged(socket: &mut WebSocket, missed: u64) {
    tracing::warn!(missed, "page fell behind; disconnecting it");
    close(socket, format!("fell behind by {missed} updates; reconnect for a fresh scene")).await;
}

/// What a page asks of the server.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PageRequest {
    clouds: bool,
}

fn javascript(source: &'static str) -> Response {
    ([(CONTENT_TYPE, "text/javascript; charset=utf-8")], source).into_response()
}

pub fn router(shared: SharedView) -> Router {
    Router::new()
        .route("/", get(|| async { Html(PAGE) }))
        .route("/three/three.module.js", get(|| async { javascript(THREE_MODULE) }))
        .route("/three/addons/controls/OrbitControls.js", get(|| async { javascript(ORBIT_CONTROLS) }))
        .route("/ws", get(socket))
        .with_state(shared)
}

/// Why an upgrade is refused: the page's WebSocket opens only from the page's own origin, so another site a viewer
/// has open cannot read the map through their browser.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum OriginRefusal {
    #[error("the upgrade has no {0} header")]
    Missing(&'static str),
    #[error("the {0} header is not text")]
    NotText(&'static str),
    #[error("origin {origin} is not this page's host {host}")]
    OtherOrigin { origin: String, host: String },
}

/// Whether the upgrade comes from a page served by this host: its origin, without the scheme, is the request's host.
pub fn check_same_origin(headers: &HeaderMap) -> Result<(), OriginRefusal> {
    let header = |name: &'static str, key| {
        headers.get(&key).ok_or(OriginRefusal::Missing(name))?.to_str().map_err(|_| OriginRefusal::NotText(name))
    };
    let (origin, host) = (header("Origin", ORIGIN)?, header("Host", HOST)?);
    let origin_host = origin.strip_prefix("http://").or_else(|| origin.strip_prefix("https://"));
    if origin_host == Some(host) {
        Ok(())
    } else {
        Err(OriginRefusal::OtherOrigin { origin: origin.to_owned(), host: host.to_owned() })
    }
}

async fn socket(upgrade: WebSocketUpgrade, headers: HeaderMap, State(shared): State<SharedView>) -> Response {
    if let Err(refusal) = check_same_origin(&headers) {
        tracing::warn!(%refusal, "refusing a WebSocket");
        return (StatusCode::FORBIDDEN, refusal.to_string()).into_response();
    }
    upgrade.on_upgrade(move |socket| serve_page(socket, shared))
}

fn message(outgoing: Outgoing) -> Message {
    match outgoing.payload {
        Payload::Binary(bytes) => Message::Binary(bytes),
        Payload::Json(text) => Message::Text(text.as_ref().into()),
    }
}

/// Why a page can no longer be sent to.
enum SendFailure {
    /// The connection is gone.
    Gone,
    /// The page did not take a message within [`SEND_DEADLINE`].
    Stalled,
}

async fn send(socket: &mut WebSocket, message: Message) -> Result<(), SendFailure> {
    match tokio::time::timeout(SEND_DEADLINE, socket.send(message)).await {
        Ok(Ok(())) => Ok(()),
        Ok(Err(_)) => Err(SendFailure::Gone),
        Err(_) => Err(SendFailure::Stalled),
    }
}

/// Sends a close frame with the reason; a page that does not take it within the deadline is dropped all the same.
async fn close(socket: &mut WebSocket, reason: String) {
    let frame = Message::Close(Some(CloseFrame { code: CLOSE_POLICY, reason: reason.into() }));
    let _ = send(socket, frame).await;
}

/// Sends every message in order; false once the page can no longer be sent to, after closing a stalled page.
async fn send_each(socket: &mut WebSocket, messages: impl IntoIterator<Item = Outgoing>) -> bool {
    for outgoing in messages {
        match send(socket, message(outgoing)).await {
            Ok(()) => {}
            Err(SendFailure::Gone) => return false,
            Err(SendFailure::Stalled) => {
                tracing::warn!(deadline_s = SEND_DEADLINE.as_secs(), "page stopped reading; disconnecting it");
                let reason = format!("took no message for {} s; reconnect for a fresh scene", SEND_DEADLINE.as_secs());
                close(socket, reason).await;
                return false;
            }
        }
    }
    true
}

async fn serve_page(mut socket: WebSocket, shared: SharedView) {
    let (snapshot, mut updates) = shared.connect();
    tracing::info!(messages = snapshot.len(), "page connected");
    if !send_each(&mut socket, snapshot).await {
        return;
    }
    let mut clouds: Option<CloudFeed> = None;
    loop {
        tokio::select! {
            // The scene first, so the robot and the people never wait behind the clouds.
            biased;
            update = updates.recv() => match update {
                Ok(outgoing) => {
                    if !send_each(&mut socket, [outgoing]).await {
                        return;
                    }
                }
                Err(broadcast::error::RecvError::Lagged(missed)) => return on_lagged(&mut socket, missed).await,
                Err(broadcast::error::RecvError::Closed) => return,
            },
            received = socket.recv() => match received {
                None | Some(Err(_)) | Some(Ok(Message::Close(_))) => {
                    tracing::info!("page disconnected");
                    return;
                }
                Some(Ok(Message::Text(text))) => match serde_json::from_str::<PageRequest>(&text) {
                    Ok(request) => match (request.clouds, clouds.is_some()) {
                        (true, false) => clouds = Some(shared.subscribe_clouds()),
                        (false, true) => clouds = None,
                        _ => {}
                    },
                    Err(error) => {
                        tracing::warn!(%error, "page sent a request the server does not know; disconnecting it");
                        close(&mut socket, format!("unknown request: {error}")).await;
                        return;
                    }
                },
                Some(Ok(_)) => {}
            },
            cloud = next_cloud(&mut clouds) => match cloud {
                Ok(outgoing) => {
                    if !send_each(&mut socket, [outgoing]).await {
                        return;
                    }
                }
                Err(broadcast::error::RecvError::Lagged(missed)) => return on_lagged(&mut socket, missed).await,
                Err(broadcast::error::RecvError::Closed) => return,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use axum::http::HeaderValue;

    use super::*;

    fn headers(origin: Option<&'static str>, host: &'static str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(HOST, HeaderValue::from_static(host));
        if let Some(origin) = origin {
            headers.insert(ORIGIN, HeaderValue::from_static(origin));
        }
        headers
    }

    #[tokio::test]
    async fn a_page_turning_the_clouds_on_gets_the_clouds_so_far_then_every_later_cloud_update_once() {
        use crate::clouds::{KeyframeMapPose, MapPoseStanding};
        use crate::cloud::CloudSettings;
        use perception_wire::localization::{FrameId, KeyframeSelection, Pose4Dof};

        let shared = SharedView::new(Scene::new(), CloudStore::new(CloudSettings::DEFAULT, 10));
        let at = |x_m: f32| KeyframeMapPose { pose: Pose4Dof { x_m, y_m: 0.0, z_m: 1.0, yaw_rad: 0.0 }, standing: MapPoseStanding::Registered, selection: KeyframeSelection::Selected };
        let keyframe = |frame_id: u64| crate::clouds::PreparedKeyframe::of(&crate::cloud::tests::wall_keyframe(frame_id, 3000, [0; 3]), &CloudSettings::DEFAULT).unwrap();
        shared.apply_clouds(|clouds| {
            let mut out = clouds.on_keyframe(keyframe(1));
            out.extend(clouds.set_map_pose(FrameId(1), at(0.0)));
            out
        });
        let mut feed = shared.subscribe_clouds();
        assert_eq!(feed.so_far.len(), 2, "the first cloud and its pose");
        shared.apply_clouds(|clouds| {
            let mut out = clouds.on_keyframe(keyframe(2));
            out.extend(clouds.set_map_pose(FrameId(2), at(1.0)));
            out
        });
        let mut kinds = Vec::new();
        for _ in 0..4 {
            kinds.push(match feed.next().await.unwrap().payload {
                Payload::Binary(bytes) => format!("cloud {}", bytes[0]),
                Payload::Json(text) => serde_json::from_str::<serde_json::Value>(&text).unwrap()["frame_id"].to_string(),
            });
        }
        assert_eq!(kinds, ["cloud 1", "1", "cloud 1", "2"], "the first cloud and pose, then the second, each once");
        assert!(feed.updates.try_recv().is_err(), "nothing more");
    }

    #[test]
    fn only_the_pages_own_origin_may_open_the_websocket() {
        assert_eq!(check_same_origin(&headers(Some("http://strixen:8094"), "strixen:8094")), Ok(()));
        assert_eq!(check_same_origin(&headers(Some("https://strixen:8094"), "strixen:8094")), Ok(()));
        assert!(matches!(
            check_same_origin(&headers(Some("http://evil.example"), "strixen:8094")),
            Err(OriginRefusal::OtherOrigin { .. })
        ));
        assert!(matches!(
            check_same_origin(&headers(Some("http://strixen:8095"), "strixen:8094")),
            Err(OriginRefusal::OtherOrigin { .. })
        ));
        assert_eq!(check_same_origin(&headers(None, "strixen:8094")), Err(OriginRefusal::Missing("Origin")));
    }
}
