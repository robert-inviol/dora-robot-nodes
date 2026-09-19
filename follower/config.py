"""Follower settings parsed from a TOML file."""

import tomllib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .drive import FULL_POWER_PCT, TrackSpeed
from .pursuit import PursuitGains
from .target import SelectionRule


class ConfigError(ValueError):
    pass


class TankMode(Enum):
    DRY_RUN = "dry_run"
    ARMED = "armed"


@dataclass(frozen=True)
class TankSettings:
    mode: TankMode
    serial_device: str


@dataclass(frozen=True)
class TargetSettings:
    selection_rule: SelectionRule
    min_confidence: float
    release_after_s: float


@dataclass(frozen=True)
class FollowerConfig:
    tank: TankSettings
    target: TargetSettings
    gains: PursuitGains
    stale_detections_after_s: float


def _within(name: str, value: float, low: float, high: float) -> float:
    if not low <= value <= high:
        raise ValueError(f"{name} is {value}, expected {low} to {high}")
    return value


def _positive(name: str, value: float) -> float:
    if value <= 0:
        raise ValueError(f"{name} is {value}, expected more than 0")
    return value


def load_config(path: Path) -> FollowerConfig:
    settings = tomllib.loads(path.read_text())
    try:
        tank, target, pursuit, safety = (
            settings["tank"],
            settings["target"],
            settings["pursuit"],
            settings["safety"],
        )
        return FollowerConfig(
            tank=TankSettings(
                mode=TankMode(tank["mode"]),
                serial_device=tank["serial_device"],
            ),
            target=TargetSettings(
                selection_rule=SelectionRule(target["selection_rule"]),
                min_confidence=_within("target.min_confidence", target["min_confidence"], 0, 1),
                release_after_s=_positive("target.release_after_s", target["release_after_s"]),
            ),
            gains=PursuitGains(
                turn=_positive("pursuit.turn_gain", pursuit["turn_gain"]),
                forward=_positive("pursuit.forward_gain", pursuit["forward_gain"]),
                target_frame_fill=_within(
                    "pursuit.target_frame_fill", pursuit["target_frame_fill"], 0, 1
                ),
                frame_fill_deadband=_within(
                    "pursuit.frame_fill_deadband", pursuit["frame_fill_deadband"], 0, 1
                ),
                max_speed=TrackSpeed(
                    int(_within("pursuit.max_speed_pct", pursuit["max_speed_pct"], 1, FULL_POWER_PCT))
                ),
            ),
            stale_detections_after_s=_positive(
                "safety.stale_detections_after_s", safety["stale_detections_after_s"]
            ),
        )
    except KeyError as missing:
        raise ConfigError(f"{path}: missing setting {missing}") from missing
    except ValueError as invalid:
        raise ConfigError(f"{path}: {invalid}") from invalid
