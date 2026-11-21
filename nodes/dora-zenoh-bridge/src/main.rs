//! Carries messages between dataflows over Zenoh. Each dora input named in ZENOH_PUBLISH is put on a key;
//! each key named in ZENOH_SUBSCRIBE comes out as a dora output as soon as Zenoh hands it over, the Zenoh samples
//! merged into the dora event stream. Each payload is one message's row as a zstd-compressed Arrow IPC record batch,
//! its wire type named by fingerprint (see `payload`); only the wire types of lib/dora-rig-messages cross. Every
//! 10 s each key's traffic is logged (see `traffic`).
//!
//! ZENOH_PUBLISH    "input=key,input=key"   e.g. "keyframe=rig/keyframe"
//! ZENOH_LATEST_ONLY "key,key"              published keys where only the newest message matters, e.g. "rig/posed_depth"
//! ZENOH_SUBSCRIBE  "key=output,key=output" e.g. "rig/keyframe=keyframe"
//! ZENOH_LISTEN     endpoints to listen on, e.g. "tcp/0.0.0.0:7447"
//! ZENOH_CONNECT    endpoints to connect to, e.g. "tcp/192.0.2.10:7447"

mod payload;
mod publish;
mod traffic;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use dora_node_api::dora_core::config::DataId;
use dora_node_api::futures::channel::mpsc;
use dora_node_api::futures::executor::block_on;
use dora_node_api::futures::StreamExt;
use dora_node_api::merged::{MergeExternal, MergedEvent};
use dora_node_api::{DoraNode, Event, MetadataParameters, Parameter};
use eyre::{eyre, Context, Result};
use zenoh::Wait;

use payload::{PayloadDecoder, PayloadEncoder};
use publish::{publish_plan, KeyPublisher};
use traffic::{capture_camera_ns, unix_now_ns, Direction, Traffic};

/// An optional timer input. Received messages never wait for it: it writes the traffic log on time when no message
/// comes, and holds the dora event stream open for a bridge that only subscribes. dora ends the event stream of a node
/// whose inputs have all closed, and such a node no longer hears `dora stop`.
const TICK_INPUT: &str = "tick";

fn pairs(variable: &str) -> Result<Vec<(String, String)>> {
    let Ok(value) = std::env::var(variable) else { return Ok(Vec::new()) };
    value
        .split(',')
        .filter(|pair| !pair.trim().is_empty())
        .map(|pair| {
            let (left, right) = pair.split_once('=').ok_or_else(|| eyre!("{variable} entry {pair:?} is not name=name"))?;
            Ok((left.trim().to_owned(), right.trim().to_owned()))
        })
        .collect()
}

fn names(variable: &str) -> Vec<String> {
    let Ok(value) = std::env::var(variable) else { return Vec::new() };
    value.split(',').map(str::trim).filter(|name| !name.is_empty()).map(str::to_owned).collect()
}

fn endpoints(variable: &str) -> String {
    let list: Vec<String> = std::env::var(variable)
        .unwrap_or_default()
        .split(',')
        .filter(|e| !e.trim().is_empty())
        .map(|e| format!("{:?}", e.trim()))
        .collect();
    format!("[{}]", list.join(","))
}

/// A Zenoh sample on a subscribed key, as the subscriber's callback hands it to the event loop.
struct Received {
    key: Arc<str>,
    output: DataId,
    payload: Vec<u8>,
    received: Instant,
    received_unix_ns: i64,
}

fn main() -> Result<()> {
    let plan = publish_plan(&pairs("ZENOH_PUBLISH")?, &names("ZENOH_LATEST_ONLY"))?;
    let subscribe = pairs("ZENOH_SUBSCRIBE")?;
    let runtime = tokio::runtime::Runtime::new()?;
    let session = runtime.block_on(async {
        let mut config = zenoh::Config::default();
        config.insert_json5("listen/endpoints", &endpoints("ZENOH_LISTEN")).map_err(|e| eyre!("ZENOH_LISTEN: {e}"))?;
        config.insert_json5("connect/endpoints", &endpoints("ZENOH_CONNECT")).map_err(|e| eyre!("ZENOH_CONNECT: {e}"))?;
        zenoh::open(config).await.map_err(|e| eyre!("opening the zenoh session: {e}"))
    })?;
    let (sender, received) = mpsc::unbounded::<Received>();
    let mut subscribers = Vec::new();
    for (key, output) in subscribe {
        let sender = sender.clone();
        let output = DataId::from(output);
        let subscribed_key: Arc<str> = Arc::from(key.as_str());
        let subscriber = runtime.block_on(async {
            session
                .declare_subscriber(key.clone())
                .callback(move |sample| {
                    let payload = sample.payload().to_bytes().into_owned();
                    let _ = sender.unbounded_send(Received {
                        key: subscribed_key.clone(),
                        output: output.clone(),
                        payload,
                        received: Instant::now(),
                        received_unix_ns: unix_now_ns(),
                    });
                })
                .await
                .map_err(|e| eyre!("subscribing to {key}: {e}"))
        })?;
        subscribers.push(subscriber);
    }
    drop(sender);
    let mut publishers: HashMap<String, (String, KeyPublisher)> = HashMap::new();
    for (input, (key, delivery)) in &plan {
        publishers.insert(input.clone(), (key.clone(), KeyPublisher::declare(&session, key, *delivery)?));
    }
    let mut traffic = Traffic::new();
    let mut encoder = PayloadEncoder::new()?;
    let mut decoder = PayloadDecoder::new()?;
    let (mut node, events) = DoraNode::init_from_env()?;
    if !subscribers.is_empty() && node.node_config().inputs.is_empty() {
        eyre::bail!("ZENOH_SUBSCRIBE names keys but the bridge has no inputs, so dora would end its event stream at once; give it a {TICK_INPUT} input, e.g. dora/timer/millis/1000");
    }
    let mut events = events.merge_external(received);

    while let Some(event) = block_on(events.next()) {
        match event {
            MergedEvent::External(message) => {
                let started = Instant::now();
                let array = decoder.decode(&message.payload).wrap_err_with(|| format!("decoding a payload received on {}", message.key))?;
                let decode = started.elapsed();
                traffic.record(Direction::Received, &message.key, message.payload.len(), decode, message.received_unix_ns, capture_camera_ns(&array));
                let mut parameters = MetadataParameters::default();
                parameters.insert("bridge_decode_ms".to_owned(), Parameter::Float(decode.as_secs_f64() * 1e3));
                parameters.insert("bridge_wait_ms".to_owned(), Parameter::Float((started - message.received).as_secs_f64() * 1e3));
                parameters.insert("bridge_bytes".to_owned(), Parameter::Integer(message.payload.len() as i64));
                node.send_output(message.output, parameters, array)?;
                traffic.log_when_due();
            }
            MergedEvent::Dora(Event::Input { id, .. }) if id.as_str() == TICK_INPUT => traffic.log_when_due(),
            MergedEvent::Dora(Event::Input { id, data, .. }) => {
                let arrival_unix_ns = unix_now_ns();
                let Some((key, publisher)) = publishers.get_mut(id.as_str()) else {
                    eyre::bail!("input {id} has no key in ZENOH_PUBLISH");
                };
                let started = Instant::now();
                let bytes = encoder.encode(data.as_ref()).wrap_err_with(|| format!("encoding input {id} for {key}"))?;
                traffic.record(Direction::Published, key, bytes.len(), started.elapsed(), arrival_unix_ns, capture_camera_ns(data.as_ref()));
                publisher.publish(bytes).wrap_err_with(|| format!("publishing input {id}"))?;
                traffic.log_when_due();
            }
            MergedEvent::Dora(Event::Stop(_)) => break,
            MergedEvent::Dora(_) => {}
        }
    }
    drop(subscribers);
    for (_, publisher) in publishers.values_mut() {
        publisher.close()?;
    }
    drop(publishers);
    session.close().wait().map_err(|e| eyre!("closing the zenoh session: {e}"))?;
    Ok(())
}
