//! The bridge's payload: a header naming its framing and the message's wire type by fingerprint, then one zstd
//! frame holding the message's single row as an Arrow IPC record batch message, without its schema.
//!
//! The receiver takes the schema from the table of wire types it was compiled with (`wire_schemas`), found by the
//! fingerprint, so no schema is ever sent and no payload depends on an earlier one. Only the registered wire types
//! cross: a message of any other type is refused on publish, and a fingerprint the table does not hold is refused on
//! receipt as coming from a peer built from different message definitions.
//!
//! The record batch message is compressed in one call through a zstd context the encoder keeps, rather than buffer
//! by buffer through Arrow IPC's own compression, which starts a zstd encoder for every buffer of every message.

use std::collections::HashMap;
use std::io::Cursor;
use std::sync::LazyLock;

use arrow_ipc::reader::read_record_batch;
use arrow_ipc::writer::{write_message, DictionaryTracker, IpcDataGenerator, IpcWriteOptions};
use arrow_ipc::{root_as_message, MessageHeader, MetadataVersion};
use dora_node_api::arrow::array::{Array, AsArray, StructArray};
use dora_node_api::arrow::buffer::Buffer;
use dora_node_api::arrow::datatypes::DataType;
use dora_node_api::arrow::record_batch::RecordBatch;
use eyre::{bail, eyre, Result, WrapErr};
use wire_schemas::{Fingerprint, WireTypeTable};
use zstd::bulk::{Compressor, Decompressor};
use zstd::zstd_safe;

/// The first bytes of every payload: "DZB" and the framing version. A payload of another framing, including the
/// plain Arrow IPC stream of an older bridge, is refused by name.
pub const PAYLOAD_MAGIC: [u8; 3] = *b"DZB";

/// The version of the framing this bridge writes and reads.
pub const FRAMING_VERSION: u8 = 2;

/// "DZB", the framing version and the wire type's fingerprint.
pub const HEADER_BYTES: usize = PAYLOAD_MAGIC.len() + 1 + wire_schemas::FINGERPRINT_BYTES;

/// The largest record batch message a payload may hold before compression. The receiver allocates what a frame says
/// it holds, so a frame naming more is refused before anything is allocated.
pub const LARGEST_MESSAGE_BYTES: usize = 64 << 20;

/// zstd's default level, the level Arrow IPC's own zstd compression uses.
const ZSTD_LEVEL: i32 = zstd::DEFAULT_COMPRESSION_LEVEL;

/// Buffers in the record batch message are padded to 8 bytes, the least the Arrow format allows.
const IPC_ALIGNMENT: usize = 8;

/// An encapsulated IPC message starts with this marker, then the metadata length as a little-endian i32.
const CONTINUATION_MARKER: [u8; 4] = [0xff; 4];
const PREFIX_BYTES: usize = 8;

/// Every wire type this bridge was compiled with.
static WIRE_TYPES: LazyLock<WireTypeTable> = LazyLock::new(WireTypeTable::new);

/// Writes messages as payloads through one zstd context and one message buffer, both reused for every message.
pub struct PayloadEncoder {
    options: IpcWriteOptions,
    generator: IpcDataGenerator,
    compressor: Compressor<'static>,
    message: Vec<u8>,
}

impl PayloadEncoder {
    pub fn new() -> Result<PayloadEncoder> {
        Ok(PayloadEncoder {
            options: IpcWriteOptions::try_new(IPC_ALIGNMENT, false, MetadataVersion::V5)?,
            generator: IpcDataGenerator::default(),
            compressor: Compressor::new(ZSTD_LEVEL).wrap_err("starting the zstd compressor")?,
            message: Vec::new(),
        })
    }

    pub fn encode(&mut self, message: &dyn Array) -> Result<Vec<u8>> {
        if !matches!(message.data_type(), DataType::Struct(_)) {
            bail!("only struct messages can cross the bridge, not {:?}", message.data_type());
        }
        let Some(wire_type) = WIRE_TYPES.by_data_type(message.data_type()) else {
            bail!(
                "the message is not a registered wire type: only the messages of lib/dora-rig-messages, as the wire-schemas crate defines them, cross the bridge; its Arrow type is {:?}",
                message.data_type()
            );
        };
        let batch = RecordBatch::from(message.as_struct().clone());
        let (dictionaries, encoded) = self.generator.encoded_batch(&batch, &mut DictionaryTracker::new(false), &self.options)?;
        if !dictionaries.is_empty() {
            bail!("the message holds dictionary arrays, which no wire type has");
        }
        self.message.clear();
        write_message(&mut self.message, encoded, &self.options)?;
        if self.message.len() > LARGEST_MESSAGE_BYTES {
            bail!("the message is {} bytes as an Arrow IPC record batch, more than the {LARGEST_MESSAGE_BYTES} a payload may hold", self.message.len());
        }
        let mut payload = Cursor::new(Vec::with_capacity(HEADER_BYTES + zstd_safe::compress_bound(self.message.len())));
        payload.get_mut().extend_from_slice(&PAYLOAD_MAGIC);
        payload.get_mut().push(FRAMING_VERSION);
        payload.get_mut().extend_from_slice(&wire_type.fingerprint.0);
        payload.set_position(HEADER_BYTES as u64);
        self.compressor.compress_to_buffer(&self.message, &mut payload).wrap_err("compressing the message")?;
        Ok(payload.into_inner())
    }
}

/// Reads payloads through one zstd context, reused for every message.
pub struct PayloadDecoder {
    decompressor: Decompressor<'static>,
}

impl PayloadDecoder {
    pub fn new() -> Result<PayloadDecoder> {
        Ok(PayloadDecoder { decompressor: Decompressor::new().wrap_err("starting the zstd decompressor")? })
    }

    /// The message a payload holds. The arrays share the decompressed message's memory.
    pub fn decode(&mut self, payload: &[u8]) -> Result<StructArray> {
        let (fingerprint, frame) = framed_message(payload)?;
        let Some(wire_type) = WIRE_TYPES.by_fingerprint(&fingerprint) else {
            bail!("the payload's wire type fingerprint {fingerprint} is not one this bridge knows: the peer was built from different message definitions");
        };
        let message_bytes = match zstd_safe::get_frame_content_size(frame) {
            Ok(Some(size)) => size,
            Ok(None) => bail!("the payload's zstd frame does not say how many bytes it holds"),
            Err(_) => bail!("the payload's body is not a zstd frame"),
        };
        if message_bytes > LARGEST_MESSAGE_BYTES as u64 {
            bail!("the payload's zstd frame holds {message_bytes} bytes, more than the {LARGEST_MESSAGE_BYTES} a payload may hold");
        }
        let mut message = Vec::with_capacity(message_bytes as usize);
        self.decompressor.decompress_to_buffer(frame, &mut message).wrap_err("decompressing the payload")?;
        if message.len() as u64 != message_bytes {
            bail!("the payload's zstd frame gave {} bytes, not the {message_bytes} it names", message.len());
        }
        let batch = read_row(Buffer::from_vec(message), wire_type)
            .wrap_err_with(|| format!("reading the payload as a {} message", wire_type.names.join(" or ")))?;
        Ok(StructArray::from(batch))
    }
}

/// The record batch an encapsulated IPC record batch message holds, read with the wire type's schema.
fn read_row(message: Buffer, wire_type: &wire_schemas::RegisteredType) -> Result<RecordBatch> {
    if message.len() < PREFIX_BYTES || message[..4] != CONTINUATION_MARKER {
        bail!("the message does not start with an Arrow IPC continuation marker");
    }
    let metadata_bytes = i32::from_le_bytes(message[4..8].try_into().expect("four bytes"));
    let metadata_end = usize::try_from(metadata_bytes).ok().and_then(|bytes| PREFIX_BYTES.checked_add(bytes)).filter(|&end| end <= message.len());
    let Some(metadata_end) = metadata_end else {
        bail!("the message's metadata is {metadata_bytes} bytes, more than the {} bytes after its prefix", message.len() - PREFIX_BYTES);
    };
    let metadata = root_as_message(&message[PREFIX_BYTES..metadata_end]).map_err(|e| eyre!("the message's metadata is not an Arrow IPC message: {e}"))?;
    if metadata.header_type() != MessageHeader::RecordBatch {
        bail!("the message is an Arrow IPC {:?} message, not a record batch", metadata.header_type());
    }
    let batch = metadata.header_as_record_batch().ok_or_else(|| eyre!("the record batch message has no record batch"))?;
    let body_bytes = message.len() - metadata_end;
    if usize::try_from(metadata.bodyLength()).ok() != Some(body_bytes) {
        bail!("the record batch names a {} byte body, and {body_bytes} bytes follow its metadata", metadata.bodyLength());
    }
    let body = message.slice(metadata_end);
    Ok(read_record_batch(&body, batch, wire_type.schema.clone(), &HashMap::new(), None, &metadata.version())?)
}

/// The fingerprint and zstd frame after the payload's header, once the header is this bridge's framing.
fn framed_message(payload: &[u8]) -> Result<(Fingerprint, &[u8])> {
    let Some((header, frame)) = payload.split_first_chunk::<HEADER_BYTES>() else {
        bail!("the payload is {} bytes, shorter than the {HEADER_BYTES}-byte header", payload.len());
    };
    if header[..3] != PAYLOAD_MAGIC {
        bail!("the payload does not start with the bridge's header \"DZB\" (it starts {:02x?}): it is not from a dora-zenoh-bridge of this framing", &header[..4]);
    }
    if header[3] != FRAMING_VERSION {
        bail!("the payload is in bridge framing version {}, and this bridge reads only version {FRAMING_VERSION}", header[3]);
    }
    let fingerprint = Fingerprint(header[4..].try_into().expect("the header's last 8 bytes"));
    Ok((fingerprint, frame))
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use arrow_ipc::reader::FileReader;
    use arrow_ipc::writer::StreamWriter;
    use arrow_ipc::CompressionType;
    use dora_node_api::arrow::array::{ArrayRef, BinaryArray, Float32Array, Int64Array, ListArray, UInt16Array, UInt64Array};
    use dora_node_api::arrow::buffer::OffsetBuffer;
    use dora_node_api::arrow::datatypes::{Field, Schema};

    use super::*;

    fn fixture_directory() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../lib/dora-rig-messages/dora_rig_messages/fixtures")
    }

    fn read_fixture(path: &PathBuf) -> StructArray {
        let mut reader = FileReader::try_new(File::open(path).unwrap(), None).unwrap();
        StructArray::from(reader.next().unwrap().unwrap())
    }

    fn fixture(name: &str) -> StructArray {
        read_fixture(&fixture_directory().join(format!("{name}.arrow")))
    }

    fn one_list(values: ArrayRef) -> ArrayRef {
        let item = Arc::new(Field::new("item", values.data_type().clone(), false));
        Arc::new(ListArray::new(item, OffsetBuffer::from_lengths([values.len()]), values, None))
    }

    /// A keyframe of 100,000 depth pixels, built on the registered keyframe type.
    fn large_keyframe(frame_id: u64) -> StructArray {
        let depth = UInt16Array::from((0..100_000).map(|i| 1000 + (i % 640) as u16).collect::<Vec<_>>());
        let float = |value: f32| Arc::new(Float32Array::from(vec![value])) as ArrayRef;
        let columns: Vec<ArrayRef> = vec![
            Arc::new(UInt64Array::from(vec![frame_id])),
            Arc::new(Int64Array::from(vec![1_762_560_000_000_000_000])),
            Arc::new(UInt16Array::from(vec![400])),
            Arc::new(UInt16Array::from(vec![250])),
            float(640.0),
            float(640.0),
            float(200.0),
            float(125.0),
            one_list(Arc::new(Float32Array::from(vec![0.0, -9.75, 0.0]))),
            one_list(Arc::new(depth)),
            Arc::new(BinaryArray::from_vec(vec![&[0xFF, 0xD8, 0xFF, 0xD9][..]])),
        ];
        StructArray::new(wire_schemas::perception::keyframe_fields(), columns, None)
    }

    /// A message whose type no wire type has.
    fn unregistered_message() -> StructArray {
        StructArray::from(vec![(Arc::new(Field::new("frame_id", DataType::UInt64, true)), Arc::new(UInt64Array::from(vec![7])) as ArrayRef)])
    }

    fn stream_with_schema(message: &StructArray, options: IpcWriteOptions) -> Vec<u8> {
        let batch = RecordBatch::from(message.clone());
        let mut bytes = Vec::new();
        let mut writer = StreamWriter::try_new_with_options(&mut bytes, &Schema::new(batch.schema().fields().clone()), options).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();
        drop(writer);
        bytes
    }

    /// A message as a plain Arrow IPC stream with Arrow's per-buffer zstd compression: the payload of the oldest
    /// bridges.
    fn per_buffer_zstd_stream(message: &StructArray) -> Vec<u8> {
        stream_with_schema(message, IpcWriteOptions::default().try_with_compression(Some(CompressionType::ZSTD)).unwrap())
    }

    /// A message in framing version 1: "DZB", 1, then one zstd frame of the Arrow IPC stream, schema included.
    fn framing_1_payload(message: &StructArray) -> Vec<u8> {
        let stream = stream_with_schema(message, IpcWriteOptions::try_new(IPC_ALIGNMENT, false, MetadataVersion::V5).unwrap());
        let mut payload = b"DZB\x01".to_vec();
        payload.extend(zstd::bulk::compress(&stream, ZSTD_LEVEL).unwrap());
        payload
    }

    fn mean_time(mut work: impl FnMut()) -> Duration {
        work();
        let started = Instant::now();
        for _ in 0..50 {
            work();
        }
        started.elapsed() / 50
    }

    #[test]
    fn every_golden_fixture_crosses_unchanged_with_no_earlier_message() {
        let mut fixtures: Vec<PathBuf> = std::fs::read_dir(fixture_directory()).unwrap().map(|entry| entry.unwrap().path()).filter(|path| path.extension().is_some_and(|e| e == "arrow")).collect();
        fixtures.sort();
        assert!(fixtures.len() > 30, "{} fixtures", fixtures.len());
        let mut encoder = PayloadEncoder::new().unwrap();
        for path in &fixtures {
            let message = read_fixture(path);
            let payload = encoder.encode(&message).unwrap_or_else(|e| panic!("{}: {e:#}", path.display()));
            let fresh_decoder_result = PayloadDecoder::new().unwrap().decode(&payload).unwrap();
            assert_eq!(fresh_decoder_result, message, "{}", path.display());
        }
    }

    #[test]
    fn one_decoder_reads_messages_of_different_types_in_any_order() {
        let mut encoder = PayloadEncoder::new().unwrap();
        let mut decoder = PayloadDecoder::new().unwrap();
        for message in ["odom_pose", "keyframe", "odom_pose", "imu", "keyframe"].map(fixture) {
            assert_eq!(decoder.decode(&encoder.encode(&message).unwrap()).unwrap(), message);
        }
    }

    #[test]
    fn large_messages_cross_unchanged_and_smaller_through_one_encoder() {
        let mut encoder = PayloadEncoder::new().unwrap();
        let mut decoder = PayloadDecoder::new().unwrap();
        for frame_id in [7, 8] {
            let payload = encoder.encode(&large_keyframe(frame_id)).unwrap();
            assert_eq!(decoder.decode(&payload).unwrap(), large_keyframe(frame_id));
            assert!(payload.len() < 200_000 / 4, "{} bytes", payload.len());
            assert!(payload.len() < framing_1_payload(&large_keyframe(frame_id)).len(), "{} bytes", payload.len());
        }
    }

    #[test]
    fn a_message_that_is_not_a_registered_wire_type_is_refused_on_publish() {
        let error = PayloadEncoder::new().unwrap().encode(&unregistered_message()).unwrap_err().to_string();
        assert!(error.contains("not a registered wire type"), "{error}");
        assert!(error.contains("frame_id"), "{error}");
    }

    #[test]
    fn a_message_that_is_not_a_struct_is_refused() {
        let error = PayloadEncoder::new().unwrap().encode(&UInt64Array::from(vec![1])).unwrap_err().to_string();
        assert!(error.contains("only struct messages"), "{error}");
    }

    #[test]
    fn a_payload_of_an_unknown_wire_type_is_refused_as_from_other_definitions() {
        let mut payload = PayloadEncoder::new().unwrap().encode(&fixture("odom_pose")).unwrap();
        payload[PAYLOAD_MAGIC.len() + 1] ^= 0xff;
        let error = PayloadDecoder::new().unwrap().decode(&payload).unwrap_err().to_string();
        assert!(error.contains("built from different message definitions"), "{error}");
    }

    #[test]
    fn a_payload_of_another_framing_is_refused_by_name() {
        let message = fixture("odom_pose");
        let mut decoder = PayloadDecoder::new().unwrap();
        let oldest_bridge = decoder.decode(&per_buffer_zstd_stream(&message)).unwrap_err().to_string();
        assert!(oldest_bridge.contains("does not start with the bridge's header"), "{oldest_bridge}");

        let framing_1 = decoder.decode(&framing_1_payload(&message)).unwrap_err().to_string();
        assert!(framing_1.contains("framing version 1, and this bridge reads only version 2"), "{framing_1}");

        let mut payload = PayloadEncoder::new().unwrap().encode(&message).unwrap();
        payload[3] = FRAMING_VERSION + 1;
        let other_version = decoder.decode(&payload).unwrap_err().to_string();
        assert!(other_version.contains(&format!("framing version {}", FRAMING_VERSION + 1)), "{other_version}");

        let short = decoder.decode(b"DZB\x02").unwrap_err().to_string();
        assert!(short.contains("shorter than the 12-byte header"), "{short}");
    }

    #[test]
    fn a_damaged_payload_is_refused() {
        let message = fixture("odom_pose");
        let payload = PayloadEncoder::new().unwrap().encode(&message).unwrap();
        let mut decoder = PayloadDecoder::new().unwrap();
        let header = &payload[..HEADER_BYTES];

        let mut not_zstd = header.to_vec();
        not_zstd.extend_from_slice(&[0u8; 32]);
        assert!(decoder.decode(&not_zstd).unwrap_err().to_string().contains("not a zstd frame"));
        assert!(decoder.decode(&payload[..payload.len() - 5]).is_err());
        let mut flipped = payload.clone();
        let last = flipped.len() - 1;
        flipped[last] ^= 0xff;
        assert!(decoder.decode(&flipped).is_err());

        let compressed = |bytes: &[u8]| [header, &zstd::bulk::compress(bytes, ZSTD_LEVEL).unwrap()].concat();
        let not_ipc = format!("{:#}", decoder.decode(&compressed(&[0x11; 64])).unwrap_err());
        assert!(not_ipc.contains("continuation marker"), "{not_ipc}");
        let mut encoder = PayloadEncoder::new().unwrap();
        encoder.encode(&message).unwrap();
        let whole = encoder.message.clone();
        let cut_body = format!("{:#}", decoder.decode(&compressed(&whole[..whole.len() - 8])).unwrap_err());
        assert!(cut_body.contains("names a"), "{cut_body}");
        let mut bad_metadata = whole.clone();
        bad_metadata[4..8].copy_from_slice(&(whole.len() as i32).to_le_bytes());
        let overlong = format!("{:#}", decoder.decode(&compressed(&bad_metadata)).unwrap_err());
        assert!(overlong.contains("more than the"), "{overlong}");
        let schema_message = stream_with_schema(&message, IpcWriteOptions::default());
        let schema_only = format!("{:#}", decoder.decode(&compressed(&schema_message)).unwrap_err());
        assert!(schema_only.contains("not a record batch") || schema_only.contains("names a"), "{schema_only}");
    }

    #[test]
    fn a_row_of_one_wire_type_sent_under_another_fingerprint_is_refused() {
        let mut payload = PayloadEncoder::new().unwrap().encode(&fixture("odom_pose")).unwrap();
        let imu = WIRE_TYPES.by_data_type(fixture("imu").data_type()).unwrap().fingerprint;
        payload[4..HEADER_BYTES].copy_from_slice(&imu.0);
        let error = format!("{:#}", PayloadDecoder::new().unwrap().decode(&payload).unwrap_err());
        assert!(error.contains("as a imu message"), "{error}");
    }

    /// Prints, for the messages the camera machine publishes, the payload size and the mean encode and decode time
    /// of this framing next to framing version 1, which carried the schema in every payload.
    #[test]
    fn payload_sizes_and_times_are_printed_next_to_the_framing_with_schema() {
        let mut encoder = PayloadEncoder::new().unwrap();
        let mut decoder = PayloadDecoder::new().unwrap();
        println!("{:<24} {:>15} {:>15} {:>12} {:>12} {:>12}", "fixture", "with schema B", "payload B", "with sch enc", "encode", "decode");
        for name in ["odom_pose", "people_tracks_in_odom", "posed_depth_frame", "keyframe", "camera_calibration", "imu", "keyframe_pose_tracking", "mesh_blocks"] {
            let message = fixture(name);
            let with_schema = framing_1_payload(&message);
            let payload = encoder.encode(&message).unwrap();
            let with_schema_encode = mean_time(|| drop(framing_1_payload(&message)));
            let encode = mean_time(|| drop(encoder.encode(&message).unwrap()));
            let decode = mean_time(|| drop(decoder.decode(&payload).unwrap()));
            let ms = |time: Duration| format!("{:.3} ms", time.as_secs_f64() * 1e3);
            println!("{name:<24} {:>15} {:>15} {:>12} {:>12} {:>12}", with_schema.len(), payload.len(), ms(with_schema_encode), ms(encode), ms(decode));
        }
    }
}
