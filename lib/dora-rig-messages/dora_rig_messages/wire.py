"""Moving one message in or out of the single-row Arrow array that a dora output carries, and the fingerprint that
names a message's type on the bridge.

A wire type's fingerprint is the first 8 bytes of the SHA-256 of its canonical encoding, ASCII text that spells out
the whole type, fields in order:

    type  = "bool" | "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" | "f32" | "f64" | "utf8" | "binary"
          | "list<" field ">" | "struct<" [ field { "," field } ] ">"
    field = name ( "?" | "!" ) ":" type        "?" may be null, "!" never null
    name  = one or more of A-Z a-z 0-9 _

A list's item field is spelt like any other, name included. Any other Arrow type, a field name outside that alphabet,
and field metadata have no canonical encoding. The wire-schemas crate in nodes/dora-depth-perception spells the same
encoding from the Rust definitions.
"""

import hashlib
import re
from typing import Any, Mapping

import pyarrow as pa

FINGERPRINT_BYTES = 8

_PRIMITIVE_NAMES = {
    pa.bool_(): "bool",
    pa.int8(): "i8",
    pa.int16(): "i16",
    pa.int32(): "i32",
    pa.int64(): "i64",
    pa.uint8(): "u8",
    pa.uint16(): "u16",
    pa.uint32(): "u32",
    pa.uint64(): "u64",
    pa.float32(): "f32",
    pa.float64(): "f64",
    pa.string(): "utf8",
    pa.binary(): "binary",
}

_FIELD_NAME = re.compile(r"[A-Za-z0-9_]+")


def one_row(row: Mapping[str, Any], wire_type: pa.StructType) -> pa.Array:
    return pa.array([row], type=wire_type)


def only_row(array: pa.Array, wire_type: pa.StructType) -> dict[str, Any]:
    if array.type != wire_type:
        raise ValueError(f"expected wire type {wire_type}, received {array.type}")
    if len(array) != 1:
        raise ValueError(f"expected one row of {wire_type}, received {len(array)}")
    return array[0].as_py()


def canonical_encoding(data_type: pa.DataType) -> str:
    if pa.types.is_struct(data_type):
        return "struct<" + ",".join(_canonical_field(data_type.field(i)) for i in range(data_type.num_fields)) + ">"
    if pa.types.is_list(data_type):
        return "list<" + _canonical_field(data_type.value_field) + ">"
    name = _PRIMITIVE_NAMES.get(data_type)
    if name is None:
        raise ValueError(f"the Arrow type {data_type} has no canonical encoding")
    return name


def _canonical_field(field: pa.Field) -> str:
    if not _FIELD_NAME.fullmatch(field.name):
        raise ValueError(f"the field name {field.name!r} is not made of A-Z, a-z, 0-9 and _")
    if field.metadata:
        raise ValueError(f"the field {field.name!r} carries metadata")
    return f"{field.name}{'?' if field.nullable else '!'}:{canonical_encoding(field.type)}"


def fingerprint(data_type: pa.DataType) -> str:
    """The fingerprint as 16 lowercase hex digits."""
    return hashlib.sha256(canonical_encoding(data_type).encode("ascii")).digest()[:FINGERPRINT_BYTES].hex()
