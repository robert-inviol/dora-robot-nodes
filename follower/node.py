"""The follower as a dora node: builds the pilot from the config file and feeds it dataflow events."""

import os
import time
from pathlib import Path

import pyarrow as pa
from dora import Node

from messages.operator import OperatorCommand

from .config import FollowerConfig, load_config
from .pilot import Pilot
from .target import TargetLock

CONFIG_PATH_VARIABLE = "FOLLOWER_CONFIG"
HUD_SOURCE = "follower"


def build_pilot(config: FollowerConfig) -> Pilot:
    return Pilot(
        lock=TargetLock(config.target.selection_rule, config.target.release_after_s),
        gains=config.gains,
        manual_max_power=config.manual_max_power,
        min_confidence=config.target.min_confidence,
        stale_after=config.stale_after,
    )


def _hud_event(message: str) -> pa.Array:
    """One row in the shape the perception bridge lists in its live events panel."""
    return pa.array(
        [
            {
                "timestamp_ms": time.time() * 1000,
                "rule_name": HUD_SOURCE,
                "event_type": HUD_SOURCE,
                "message": message,
            }
        ]
    )


def main() -> None:
    config = load_config(Path(os.environ[CONFIG_PATH_VARIABLE]))
    print(
        f"[follower] follow max {config.gains.max_power.percent}%, "
        f"manual max {config.manual_max_power.percent}%, "
        f"rule {config.target.selection_rule.value}"
    )
    pilot = build_pilot(config)
    node = Node()
    for event in node:
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT":
            continue
        now_s = time.monotonic()
        if event["id"] == "operator":
            pilot.on_operator(OperatorCommand.from_arrow(event["value"]), now_s)
        elif event["id"] == "detections":
            announcement = pilot.on_detections(event["value"].to_pylist(), now_s)
            if announcement is not None:
                print(f"[follower] {announcement}")
                node.send_output("events", _hud_event(announcement))
        elif event["id"] == "tick":
            pilot.on_tick(now_s)
            node.send_output("status", pilot.status.to_arrow())
        # Every input refreshes the demand, which is also what keeps the driver from timing out.
        node.send_output("drive", pilot.demand.to_arrow())
