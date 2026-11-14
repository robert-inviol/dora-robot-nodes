//! Single-row messages with nested lists and structs, moved between serde types and Arrow through arrow's JSON
//! codec. The wire type is always given explicitly, so the Arrow side matches the Python definition exactly.

use std::sync::Arc;

use dora_node_api::arrow::array::{Array, StructArray};
use dora_node_api::arrow::datatypes::{DataType, Field, Fields, Schema};
use dora_node_api::arrow::json::{ArrayWriter, ReaderBuilder};
use dora_node_api::arrow::record_batch::RecordBatch;
use eyre::{eyre, Context, Result};
use serde::de::DeserializeOwned;
use serde::Serialize;

pub(crate) fn required(name: &str, data_type: DataType) -> Field {
    Field::new(name, data_type, false)
}

pub(crate) fn optional(name: &str, data_type: DataType) -> Field {
    Field::new(name, data_type, true)
}

pub(crate) fn list_of(item_type: DataType, nullable_items: bool) -> DataType {
    DataType::List(Arc::new(Field::new("item", item_type, nullable_items)))
}

pub(crate) fn floats_struct(names: &[&str]) -> DataType {
    DataType::Struct(names.iter().map(|name| required(name, DataType::Float32)).collect())
}

pub(crate) fn encode_row<T: Serialize>(row: &T, fields: Fields) -> StructArray {
    let mut decoder = ReaderBuilder::new(Arc::new(Schema::new(fields)))
        .build_decoder()
        .expect("a wire type is a valid JSON decoder schema");
    decoder.serialize(std::slice::from_ref(row)).expect("a message serializes into its own wire type");
    StructArray::from(decoder.flush().expect("a serialized message flushes").expect("one row was serialized"))
}

pub(crate) fn decode_row<T: DeserializeOwned>(array: &dyn Array, fields: &Fields, name: &str) -> Result<T> {
    let row = crate::checked(array, fields, name)?;
    let mut writer = ArrayWriter::new(Vec::new());
    writer.write(&RecordBatch::from(row.clone())).wrap_err_with(|| format!("reading a {name} message"))?;
    writer.finish().wrap_err_with(|| format!("reading a {name} message"))?;
    let mut rows: Vec<T> =
        serde_json::from_slice(&writer.into_inner()).wrap_err_with(|| format!("decoding a {name} message"))?;
    rows.pop().ok_or_else(|| eyre!("a {name} message has no row"))
}
