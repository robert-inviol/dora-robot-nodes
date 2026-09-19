"""Moving one message in or out of the single-row Arrow array that a dora output carries."""

from typing import Any, Mapping

import pyarrow as pa


def one_row(row: Mapping[str, Any], wire_type: pa.StructType) -> pa.Array:
    return pa.array([row], type=wire_type)


def only_row(array: pa.Array, wire_type: pa.StructType) -> dict[str, Any]:
    if array.type != wire_type:
        raise ValueError(f"expected wire type {wire_type}, received {array.type}")
    if len(array) != 1:
        raise ValueError(f"expected one row of {wire_type}, received {len(array)}")
    return array[0].as_py()
