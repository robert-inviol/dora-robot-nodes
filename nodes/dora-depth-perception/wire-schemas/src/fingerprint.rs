//! A wire type's fingerprint: the first 8 bytes of the SHA-256 of its canonical encoding.
//!
//! The canonical encoding is ASCII text that spells out the whole type, fields in order:
//!
//! ```text
//! type  = "bool" | "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" | "f32" | "f64" | "utf8" | "binary"
//!       | "list<" field ">" | "struct<" [ field { "," field } ] ">"
//! field = name ( "?" | "!" ) ":" type        "?" may be null, "!" never null
//! name  = one or more of A-Z a-z 0-9 _
//! ```
//!
//! A list's item field is spelt like any other, name included. Any other Arrow type, a field name outside that
//! alphabet, and field metadata have no canonical encoding. dora_rig_messages.wire spells the same encoding from a
//! pyarrow type, so both languages give one fingerprint for one definition.

use std::fmt;

use arrow_schema::{DataType, Field};
use sha2::{Digest, Sha256};

/// The bytes of the SHA-256 a fingerprint keeps.
pub const FINGERPRINT_BYTES: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Fingerprint(pub [u8; FINGERPRINT_BYTES]);

impl fmt::Display for Fingerprint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.iter().try_for_each(|byte| write!(formatter, "{byte:02x}"))
    }
}

/// Why a type has no canonical encoding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NotCanonical {
    UnsupportedType(DataType),
    FieldName(String),
    FieldMetadata(String),
}

impl fmt::Display for NotCanonical {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NotCanonical::UnsupportedType(data_type) => write!(formatter, "the Arrow type {data_type:?} has no canonical encoding"),
            NotCanonical::FieldName(name) => write!(formatter, "the field name {name:?} is not made of A-Z, a-z, 0-9 and _"),
            NotCanonical::FieldMetadata(name) => write!(formatter, "the field {name:?} carries metadata"),
        }
    }
}

impl std::error::Error for NotCanonical {}

pub fn fingerprint(data_type: &DataType) -> Result<Fingerprint, NotCanonical> {
    let digest = Sha256::digest(canonical_encoding(data_type)?.as_bytes());
    let mut bytes = [0; FINGERPRINT_BYTES];
    bytes.copy_from_slice(&digest[..FINGERPRINT_BYTES]);
    Ok(Fingerprint(bytes))
}

pub fn canonical_encoding(data_type: &DataType) -> Result<String, NotCanonical> {
    let mut text = String::new();
    write_type(data_type, &mut text)?;
    Ok(text)
}

fn write_type(data_type: &DataType, text: &mut String) -> Result<(), NotCanonical> {
    let primitive = match data_type {
        DataType::Boolean => "bool",
        DataType::Int8 => "i8",
        DataType::Int16 => "i16",
        DataType::Int32 => "i32",
        DataType::Int64 => "i64",
        DataType::UInt8 => "u8",
        DataType::UInt16 => "u16",
        DataType::UInt32 => "u32",
        DataType::UInt64 => "u64",
        DataType::Float32 => "f32",
        DataType::Float64 => "f64",
        DataType::Utf8 => "utf8",
        DataType::Binary => "binary",
        DataType::List(item) => {
            text.push_str("list<");
            write_field(item, text)?;
            text.push('>');
            return Ok(());
        }
        DataType::Struct(fields) => {
            text.push_str("struct<");
            for (index, field) in fields.iter().enumerate() {
                if index > 0 {
                    text.push(',');
                }
                write_field(field, text)?;
            }
            text.push('>');
            return Ok(());
        }
        other => return Err(NotCanonical::UnsupportedType(other.clone())),
    };
    text.push_str(primitive);
    Ok(())
}

fn write_field(field: &Field, text: &mut String) -> Result<(), NotCanonical> {
    let name = field.name();
    if name.is_empty() || !name.bytes().all(|byte| byte.is_ascii_alphanumeric() || byte == b'_') {
        return Err(NotCanonical::FieldName(name.clone()));
    }
    if !field.metadata().is_empty() {
        return Err(NotCanonical::FieldMetadata(name.clone()));
    }
    text.push_str(name);
    text.push(if field.is_nullable() { '?' } else { '!' });
    text.push(':');
    write_type(field.data_type(), text)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use arrow_schema::Fields;

    use super::*;

    #[test]
    fn a_struct_is_spelt_field_by_field_with_nullability_and_list_items() {
        let data_type = DataType::Struct(Fields::from(vec![
            Field::new("frame_id", DataType::UInt64, true),
            Field::new("depth_mm", DataType::List(Arc::new(Field::new("item", DataType::UInt16, false))), false),
        ]));
        assert_eq!(canonical_encoding(&data_type).unwrap(), "struct<frame_id?:u64,depth_mm!:list<item!:u16>>");
    }

    #[test]
    fn the_fingerprint_is_the_start_of_the_sha256_of_the_encoding() {
        let data_type = DataType::Struct(Fields::from(vec![Field::new("requested_unix_ns", DataType::Int64, true)]));
        assert_eq!(canonical_encoding(&data_type).unwrap(), "struct<requested_unix_ns?:i64>");
        let digest = Sha256::digest(b"struct<requested_unix_ns?:i64>");
        assert_eq!(fingerprint(&data_type).unwrap().0, digest[..8]);
    }

    #[test]
    fn order_nullability_and_names_all_change_the_fingerprint() {
        let struct_of = |fields: Vec<Field>| fingerprint(&DataType::Struct(Fields::from(fields))).unwrap();
        let a = || Field::new("a", DataType::Float32, true);
        let b = || Field::new("b", DataType::Float32, true);
        let original = struct_of(vec![a(), b()]);
        assert_ne!(original, struct_of(vec![b(), a()]));
        assert_ne!(original, struct_of(vec![a(), Field::new("b", DataType::Float32, false)]));
        assert_ne!(original, struct_of(vec![a(), Field::new("c", DataType::Float32, true)]));
        assert_ne!(original, struct_of(vec![a(), Field::new("b", DataType::Float64, true)]));
    }

    #[test]
    fn a_type_outside_the_encoding_has_no_fingerprint() {
        let large = DataType::Struct(Fields::from(vec![Field::new("text", DataType::LargeUtf8, true)]));
        assert_eq!(fingerprint(&large), Err(NotCanonical::UnsupportedType(DataType::LargeUtf8)));
        let spaced = DataType::Struct(Fields::from(vec![Field::new("a b", DataType::Utf8, true)]));
        assert_eq!(fingerprint(&spaced), Err(NotCanonical::FieldName("a b".to_owned())));
        let with_metadata = Field::new("a", DataType::Utf8, true).with_metadata(HashMap::from([("unit".to_owned(), "m".to_owned())]));
        assert_eq!(fingerprint(&DataType::Struct(Fields::from(vec![with_metadata]))), Err(NotCanonical::FieldMetadata("a".to_owned())));
    }

    #[test]
    fn a_fingerprint_prints_as_16_hex_digits() {
        assert_eq!(Fingerprint([0, 1, 0xab, 0xcd, 0xef, 0x10, 0x20, 0xff]).to_string(), "0001abcdef1020ff");
    }
}
