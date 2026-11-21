import hashlib

import pyarrow as pa
import pytest

from dora_rig_messages.fixtures import EXAMPLES, WIRE_TYPES, fingerprint_table, read_fingerprint_table
from dora_rig_messages.wire import canonical_encoding, fingerprint


def test_every_fingerprint_is_the_one_in_the_checked_in_table():
    assert fingerprint_table() == read_fingerprint_table()


@pytest.mark.parametrize("name", EXAMPLES)
def test_every_golden_example_is_a_registered_wire_type(name):
    assert EXAMPLES[name].to_arrow().type in WIRE_TYPES.values()


def test_distinct_wire_types_have_distinct_fingerprints():
    distinct_types = {str(wire_type) for wire_type in WIRE_TYPES.values()}

    assert len({fingerprint(wire_type) for wire_type in WIRE_TYPES.values()}) == len(distinct_types)


def test_a_struct_is_spelt_field_by_field_with_nullability_and_list_items():
    data_type = pa.struct(
        [("frame_id", pa.uint64()), pa.field("depth_mm", pa.list_(pa.field("item", pa.uint16(), nullable=False)), nullable=False)]
    )

    assert canonical_encoding(data_type) == "struct<frame_id?:u64,depth_mm!:list<item!:u16>>"


def test_the_fingerprint_is_the_start_of_the_sha256_of_the_encoding():
    data_type = pa.struct([("requested_unix_ns", pa.int64())])

    assert fingerprint(data_type) == hashlib.sha256(b"struct<requested_unix_ns?:i64>").digest()[:8].hex()


def test_order_and_nullability_change_the_fingerprint():
    original = fingerprint(pa.struct([("a", pa.float32()), ("b", pa.float32())]))

    assert original != fingerprint(pa.struct([("b", pa.float32()), ("a", pa.float32())]))
    assert original != fingerprint(pa.struct([("a", pa.float32()), pa.field("b", pa.float32(), nullable=False)]))


@pytest.mark.parametrize(
    ("data_type", "message"),
    [
        (pa.struct([("text", pa.large_string())]), "no canonical encoding"),
        (pa.struct([("a b", pa.string())]), "not made of"),
        (pa.struct([pa.field("a", pa.string(), metadata={"unit": "m"})]), "carries metadata"),
    ],
)
def test_a_type_outside_the_encoding_has_no_fingerprint(data_type, message):
    with pytest.raises(ValueError, match=message):
        fingerprint(data_type)
