"""Golden examples of every message, stored as Arrow IPC files that each language's tests read.

Regenerate with `python -m dora_rig_messages.fixtures` after a deliberate change to a wire type.
"""

from pathlib import Path
from typing import Protocol

import pyarrow as pa

from .drive import DriveDemand
from .operator import DriveMode, OperatorCommand
from .scan import RangeReading, Scan
from .status import DriverStatus, PilotStatus, TrackId

FIXTURE_DIR = Path(__file__).parent / "fixtures"


class Message(Protocol):
    def to_arrow(self) -> pa.Array: ...


# Every float is exact in float32, so a fixture reads back equal to its example.
EXAMPLES: dict[str, Message] = {
    "drive_demand": DriveDemand(forward=0.5, turn=-0.25),
    "operator_command": OperatorCommand(DriveMode.FOLLOW, throttle=0.75, steer=-0.5),
    "pilot_status": PilotStatus(DriveMode.FOLLOW, locked_id=TrackId(7)),
    "pilot_status_nobody_locked": PilotStatus(DriveMode.MANUAL, locked_id=None),
    "driver_status": DriverStatus(armed=True, left_pct=35, right_pct=-12),
    "scan": Scan(
        spin_rev_per_s=8.0,
        readings=(
            RangeReading(bearing_deg=0.0, range_m=0.5),
            RangeReading(bearing_deg=90.0, range_m=None),
            RangeReading(bearing_deg=180.0, range_m=3.25),
            RangeReading(bearing_deg=292.5, range_m=11.75),
        ),
    ),
}


def fixture_path(name: str) -> Path:
    return FIXTURE_DIR / f"{name}.arrow"


def read_fixture(name: str) -> pa.Array:
    with pa.ipc.open_file(fixture_path(name)) as reader:
        return reader.read_all().to_batches()[0].to_struct_array()


def write_fixture(name: str, message: Message) -> None:
    batch = pa.RecordBatch.from_struct_array(message.to_arrow())
    with pa.ipc.new_file(fixture_path(name), batch.schema) as writer:
        writer.write_batch(batch)


if __name__ == "__main__":
    for example_name, example in EXAMPLES.items():
        write_fixture(example_name, example)
        print(f"wrote {fixture_path(example_name)}")
