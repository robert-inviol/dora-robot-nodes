"""The follower as a dora node: builds the pilot from the config file and feeds it dataflow events."""

import os
import signal
import sys
import time
from pathlib import Path

import pyarrow as pa
from dora import Node

from .config import FollowerConfig, TankMode, TankSettings, load_config
from .operator import OperatorCommand
from .pilot import Pilot
from .tank import LoggedTank, SerialTank, Tank, open_serial_port
from .target import TargetLock

CONFIG_PATH_VARIABLE = "FOLLOWER_CONFIG"
HUD_SOURCE = "follower"


def build_tank(settings: TankSettings) -> Tank:
    if settings.mode is TankMode.ARMED:
        return SerialTank(open_serial_port(settings.serial_device))
    return LoggedTank()


def build_pilot(config: FollowerConfig, tank: Tank) -> Pilot:
    return Pilot(
        lock=TargetLock(config.target.selection_rule, config.target.release_after_s),
        gains=config.gains,
        manual_max_speed=config.manual_max_speed,
        tank=tank,
        armed=config.tank.mode is TankMode.ARMED,
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
        f"[follower] tank mode {config.tank.mode.value}, "
        f"follow max {config.gains.max_speed.percent}%, "
        f"manual max {config.manual_max_speed.percent}%, "
        f"rule {config.target.selection_rule.value}"
    )
    # SIGTERM becomes an exit so the finally block still stops the motors.
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))

    tank = build_tank(config.tank)
    pilot = build_pilot(config, tank)
    node = Node()
    try:
        for event in node:
            if event["type"] == "STOP":
                break
            if event["type"] != "INPUT":
                continue
            now_s = time.monotonic()
            if event["id"] == "operator":
                for row in event["value"].to_pylist():
                    pilot.on_operator(OperatorCommand.from_row(row), now_s)
            elif event["id"] == "detections":
                announcement = pilot.on_detections(event["value"].to_pylist(), now_s)
                if announcement is not None:
                    print(f"[follower] {announcement}")
                    node.send_output("events", _hud_event(announcement))
            elif event["id"] == "tick":
                pilot.on_tick(now_s)
                node.send_output("status", pa.array([pilot.status.to_row()]))
    finally:
        tank.stop()
