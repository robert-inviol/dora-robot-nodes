//! Carries messages between dataflows over Zenoh. Each dora input named in ZENOH_PUBLISH is put on a key;
//! each key named in ZENOH_SUBSCRIBE comes out as a dora output. Payloads are Arrow IPC streams with zstd
//! compression, so any struct message crosses intact.
//!
//! ZENOH_PUBLISH    "input=key,input=key"   e.g. "keyframe=rig/keyframe"
//! ZENOH_SUBSCRIBE  "key=output,key=output" e.g. "rig/keyframe=keyframe"
//! ZENOH_LISTEN     endpoints to listen on, e.g. "tcp/0.0.0.0:7447"
//! ZENOH_CONNECT    endpoints to connect to, e.g. "tcp/192.0.2.10:7447"

mod payload;

use std::collections::HashMap;
use std::sync::mpsc;
use std::time::Instant;

use dora_node_api::dora_core::config::DataId;
use dora_node_api::{DoraNode, Event, MetadataParameters, Parameter};
use eyre::{eyre, Context, Result};
use zenoh::Wait;

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
    output: DataId,
    payload: Vec<u8>,
    received: Instant,
}

fn main() -> Result<()> {
    let publish: HashMap<String, String> = pairs("ZENOH_PUBLISH")?.into_iter().collect();
    let subscribe = pairs("ZENOH_SUBSCRIBE")?;
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
        let subscriber = runtime.block_on(async {
            session
                .declare_subscriber(key.clone())
                .callback(move |sample| {
                    let payload = sample.payload().to_bytes().into_owned();
                    let _ = sender.send(Received { output: output.clone(), payload, received: Instant::now() });
                })
                .await
                .map_err(|e| eyre!("subscribing to {key}: {e}"))
        })?;
        subscribers.push(subscriber);
    }
    let (mut node, mut events) = DoraNode::init_from_env()?;

    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, .. } if id.as_str() == TICK_INPUT => {
                for message in receiver.try_iter() {
                    let started = Instant::now();
                    let array = payload::decode(&message.payload).wrap_err("decoding a zenoh payload")?;
                    let mut parameters = MetadataParameters::default();
                    parameters.insert("bridge_decode_ms".to_owned(), Parameter::Float(started.elapsed().as_secs_f64() * 1e3));
                    parameters.insert("bridge_wait_ms".to_owned(), Parameter::Float((started - message.received).as_secs_f64() * 1e3));
                    parameters.insert("bridge_bytes".to_owned(), Parameter::Integer(message.payload.len() as i64));
                    node.send_output(message.output, parameters, array)?;
                }
            }
            Event::Input { id, data, .. } => {
                let Some(key) = publish.get(id.as_str()) else {
                    eyre::bail!("input {id} has no key in ZENOH_PUBLISH");
                };
                let bytes = payload::encode(data.as_ref()).wrap_err_with(|| format!("encoding input {id}"))?;
                session.put(key.as_str(), bytes).wait().map_err(|e| eyre!("publishing {id} to {key}: {e}"))?;
            }
            Event::Stop(_) => break,
            _ => {}
        }
    }
    drop(subscribers);
    session.close().wait().map_err(|e| eyre!("closing the zenoh session: {e}"))?;
    Ok(())
}
