//! The page and its WebSocket. A page that connects gets the scene so far, then every update as it happens. A page
//! that falls too far behind is disconnected with a reason, and reconnects for a fresh snapshot.

use std::sync::{Arc, Mutex};

use axum::extract::ws::{CloseFrame, Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::{Html, IntoResponse};
use axum::routing::get;
use axum::Router;
use tokio::sync::broadcast;

use crate::scene::Scene;
use crate::view::Outgoing;

const PAGE: &str = include_str!("page.html");

/// Updates a page may fall behind by before it is disconnected.
const UPDATE_BACKLOG: usize = 4096;

/// WebSocket close code for a policy-driven close.
const CLOSE_POLICY: u16 = 1008;

/// The scene and the channel its updates go out on. Updates are applied and sent under the scene's lock, so a page
/// that takes a snapshot under the same lock misses none and sees none twice.
#[derive(Clone)]
pub struct SharedScene {
    scene: Arc<Mutex<Scene>>,
    updates: broadcast::Sender<Outgoing>,
}

impl SharedScene {
    pub fn new(scene: Scene) -> SharedScene {
        SharedScene { scene: Arc::new(Mutex::new(scene)), updates: broadcast::channel(UPDATE_BACKLOG).0 }
    }

    /// Changes the scene and sends the updates the change made to every connected page.
    pub fn apply<E>(&self, change: impl FnOnce(&mut Scene) -> Result<Vec<Outgoing>, E>) -> Result<(), E> {
        let mut scene = self.scene.lock().expect("a scene change panicked");
        for update in change(&mut scene)? {
            // No page connected is not an error: the update is in the scene for the next one.
            let _ = self.updates.send(update);
        }
        Ok(())
    }

    fn connect(&self) -> (Vec<Outgoing>, broadcast::Receiver<Outgoing>) {
        let scene = self.scene.lock().expect("a scene change panicked");
        (scene.snapshot(), self.updates.subscribe())
    }
}

pub fn router(shared: SharedScene) -> Router {
    Router::new().route("/", get(|| async { Html(PAGE) })).route("/ws", get(socket)).with_state(shared)
}

async fn socket(upgrade: WebSocketUpgrade, State(shared): State<SharedScene>) -> impl IntoResponse {
    upgrade.on_upgrade(move |socket| serve_page(socket, shared))
}

fn message(outgoing: Outgoing) -> Message {
    match outgoing {
        Outgoing::Binary(bytes) => Message::Binary(bytes),
        Outgoing::Json(text) => Message::Text(text.as_ref().into()),
    }
}

async fn serve_page(mut socket: WebSocket, shared: SharedScene) {
    let (snapshot, mut updates) = shared.connect();
    tracing::info!(messages = snapshot.len(), "page connected");
    for outgoing in snapshot {
        if socket.send(message(outgoing)).await.is_err() {
            return;
        }
    }
    loop {
        tokio::select! {
            update = updates.recv() => match update {
                Ok(outgoing) => {
                    if socket.send(message(outgoing)).await.is_err() {
                        return;
                    }
                }
                Err(broadcast::error::RecvError::Lagged(missed)) => {
                    tracing::warn!(missed, "page fell behind; disconnecting it");
                    let reason = format!("fell behind by {missed} updates; reconnect for a fresh scene");
                    let close = CloseFrame { code: CLOSE_POLICY, reason: reason.into() };
                    let _ = socket.send(Message::Close(Some(close))).await;
                    return;
                }
                Err(broadcast::error::RecvError::Closed) => return,
            },
            received = socket.recv() => match received {
                None | Some(Err(_)) | Some(Ok(Message::Close(_))) => {
                    tracing::info!("page disconnected");
                    return;
                }
                Some(Ok(_)) => {}
            },
        }
    }
}
