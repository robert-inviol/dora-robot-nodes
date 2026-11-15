//! Carries messages between dataflows over Zenoh. Each dora input named in ZENOH_PUBLISH is put on a key;
//! each key named in ZENOH_SUBSCRIBE comes out as a dora output. Payloads are Arrow IPC streams with zstd
//! compression, so any struct message crosses intact. Every 10 s each key's traffic is logged (see `traffic`).
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
use std::sync::{mpsc, Arc};
use std::time::Instant;

use dora_node_api::dora_core::config::DataId;
use dora_node_api::{DoraNode, Event, MetadataParameters, Parameter};
use eyre::{eyre, Context, Result};
use zenoh::Wait;

use payload::PayloadEncoder;
use publish::{publish_plan, KeyPublisher};
use traffic::{capture_camera_ns, unix_now_ns, Direction, Traffic};

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
    let subscribes = !subscribe.is_empty();
    let runtime = tokio::runtime::Runtime::new()?;
    let session = runtime.block_on(async {
        let mut config = zenoh::Config::default();
        config.insert_json5("listen/endpoints", &endpoints("ZENOH_LISTEN")).map_err(|e| eyre!("ZENOH_LISTEN: {e}"))?;
        config.insert_json5("connect/endpoints", &endpoints("ZENOH_CONNECT")).map_err(|e| eyre!("ZENOH_CONNECT: {e}"))?;
        zenoh::open(config).await.map_err(|e| eyre!("opening the zenoh session: {e}"))
    })?;
    let (sender, receiver) = mpsc::channel::<Received>();
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
                    let _ = sender.send(Received {
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
    let mut publishers: HashMap<String, (String, KeyPublisher)> = HashMap::new();
    for (input, (key, delivery)) in &plan {
        publishers.insert(input.clone(), (key.clone(), KeyPublisher::declare(&session, key, *delivery)?));
    }
    let mut traffic = Traffic::new();
    let encoder = PayloadEncoder::new()?;
    let (mut node, mut events) = DoraNode::init_from_env()?;
    if subscribes && !node.node_config().inputs.contains_key(&DataId::from(TICK_INPUT.to_owned())) {
        eyre::bail!("ZENOH_SUBSCRIBE names keys but there is no {TICK_INPUT} input to hand their messages to dora on");
    }

    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, .. } if id.as_str() == TICK_INPUT => {
                for message in receiver.try_iter() {
                    let started = Instant::now();
                    let array = payload::decode(&message.payload).wrap_err("decoding a zenoh payload")?;
                    let decode = started.elapsed();
                    traffic.record(Direction::Received, &message.key, message.payload.len(), decode, message.received_unix_ns, capture_camera_ns(&array));
                    let mut parameters = MetadataParameters::default();
                    parameters.insert("bridge_decode_ms".to_owned(), Parameter::Float(decode.as_secs_f64() * 1e3));
                    parameters.insert("bridge_wait_ms".to_owned(), Parameter::Float((started - message.received).as_secs_f64() * 1e3));
                    parameters.insert("bridge_bytes".to_owned(), Parameter::Integer(message.payload.len() as i64));
                    node.send_output(message.output, parameters, array)?;
                }
                traffic.log_when_due();
            }
            Event::Input { id, data, .. } => {
                let arrival_unix_ns = unix_now_ns();
                let Some((key, publisher)) = publishers.get_mut(id.as_str()) else {
                    eyre::bail!("input {id} has no key in ZENOH_PUBLISH");
                };
                let started = Instant::now();
                let bytes = encoder.encode(data.as_ref()).wrap_err_with(|| format!("encoding input {id}"))?;
                traffic.record(Direction::Published, key, bytes.len(), started.elapsed(), arrival_unix_ns, capture_camera_ns(data.as_ref()));
                publisher.publish(bytes).wrap_err_with(|| format!("publishing input {id}"))?;
                traffic.log_when_due();
            }
            Event::Stop(_) => break,
            _ => {}
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
