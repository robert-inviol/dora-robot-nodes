//! A struct message as a one-batch Arrow IPC stream, compressed with zstd.

use std::sync::Arc;

use arrow_ipc::reader::StreamReader;
use arrow_ipc::writer::{IpcWriteOptions, StreamWriter};
use arrow_ipc::CompressionType;
use dora_node_api::arrow::array::{Array, AsArray, StructArray};
use dora_node_api::arrow::datatypes::{DataType, Schema};
use dora_node_api::arrow::record_batch::RecordBatch;
use eyre::{bail, Result};

pub fn encode(message: &dyn Array) -> Result<Vec<u8>> {
    if !matches!(message.data_type(), DataType::Struct(_)) {
        bail!("only struct messages can cross the bridge, not {:?}", message.data_type());
    }
    let batch = RecordBatch::from(message.as_struct().clone());
    let options = IpcWriteOptions::default().try_with_compression(Some(CompressionType::ZSTD))?;
    let mut bytes = Vec::new();
    let mut writer = StreamWriter::try_new_with_options(&mut bytes, &Schema::new(batch.schema().fields().clone()), options)?;
    writer.write(&batch)?;
    writer.finish()?;
    drop(writer);
    Ok(bytes)
}

pub fn decode(bytes: &[u8]) -> Result<StructArray> {
    let mut reader = StreamReader::try_new(bytes, None)?;
    let Some(batch) = reader.next() else { bail!("the payload holds no message") };
    let batch = batch?;
    if reader.next().is_some() {
        bail!("the payload holds more than one message");
    }
    Ok(StructArray::from(batch))
}

#[cfg(test)]
mod tests {
    use dora_node_api::arrow::array::{ArrayRef, ListArray, UInt16Array, UInt64Array};
    use dora_node_api::arrow::buffer::OffsetBuffer;
    use dora_node_api::arrow::datatypes::Field;

    use super::*;

    #[test]
    fn a_message_crosses_unchanged_and_smaller() {
        let depth = UInt16Array::from((0..100_000).map(|i| 1000 + (i % 640) as u16).collect::<Vec<_>>());
        let item = Arc::new(Field::new("item", DataType::UInt16, false));
        let depth_list = ListArray::new(item, OffsetBuffer::from_lengths([depth.len()]), Arc::new(depth), None);
        let message = StructArray::from(vec![
            (Arc::new(Field::new("frame_id", DataType::UInt64, true)), Arc::new(UInt64Array::from(vec![7])) as ArrayRef),
            (Arc::new(Field::new("depth_mm", depth_list.data_type().clone(), true)), Arc::new(depth_list) as ArrayRef),
        ]);
        let bytes = encode(&message).unwrap();
        assert_eq!(decode(&bytes).unwrap(), message);
        assert!(bytes.len() < 200_000 / 4, "{} bytes", bytes.len());
    }
}
