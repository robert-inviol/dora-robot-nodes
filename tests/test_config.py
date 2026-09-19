from pathlib import Path

import pytest

from follower.config import ConfigError, TankMode, load_config
from follower.target import SelectionRule

SHIPPED_CONFIG = Path(__file__).parent.parent / "follower.toml"


def _config_with(tmp_path: Path, old: str, new: str) -> Path:
    shipped = SHIPPED_CONFIG.read_text()
    assert old in shipped
    path = tmp_path / "follower.toml"
    path.write_text(shipped.replace(old, new))
    return path


def test_the_shipped_config_moves_nothing_until_someone_arms_it():
    assert load_config(SHIPPED_CONFIG).tank.mode is TankMode.DRY_RUN


def test_the_shipped_config_is_capped_at_the_child_safe_speed():
    assert load_config(SHIPPED_CONFIG).gains.max_speed.percent == 20


def test_the_tank_can_be_armed(tmp_path):
    path = _config_with(tmp_path, 'mode = "dry_run"', 'mode = "armed"')

    assert load_config(path).tank.mode is TankMode.ARMED


def test_the_selection_rule_is_read(tmp_path):
    path = _config_with(tmp_path, '"largest"', '"most_central"')

    assert load_config(path).target.selection_rule is SelectionRule.MOST_CENTRAL


def test_an_unknown_selection_rule_is_rejected(tmp_path):
    path = _config_with(tmp_path, 'selection_rule = "largest"', 'selection_rule = "tallest"')

    with pytest.raises(ConfigError, match="tallest"):
        load_config(path)


@pytest.mark.parametrize("max_speed_pct", [0, 101])
def test_a_speed_cap_outside_1_to_100_is_rejected(tmp_path, max_speed_pct):
    path = _config_with(tmp_path, "max_speed_pct = 20", f"max_speed_pct = {max_speed_pct}")

    with pytest.raises(ConfigError, match="pursuit.max_speed_pct"):
        load_config(path)


@pytest.mark.parametrize("max_speed_pct", [1, 100])
def test_a_speed_cap_from_1_to_100_is_accepted(tmp_path, max_speed_pct):
    path = _config_with(tmp_path, "max_speed_pct = 20", f"max_speed_pct = {max_speed_pct}")

    assert load_config(path).gains.max_speed.percent == max_speed_pct


def test_a_missing_setting_is_named(tmp_path):
    path = _config_with(tmp_path, "turn_gain = 0.8", "")

    with pytest.raises(ConfigError, match="turn_gain"):
        load_config(path)


@pytest.mark.parametrize("max_speed_pct", [0, 101])
def test_a_manual_speed_cap_outside_1_to_100_is_rejected(tmp_path, max_speed_pct):
    path = _config_with(tmp_path, "max_speed_pct = 50", f"max_speed_pct = {max_speed_pct}")

    with pytest.raises(ConfigError, match="teleop.max_speed_pct"):
        load_config(path)


def test_both_stale_input_timeouts_are_read(tmp_path):
    path = _config_with(tmp_path, "stale_operator_after_s = 0.5", "stale_operator_after_s = 0.8")

    stale_after = load_config(path).stale_after

    assert (stale_after.detections_s, stale_after.operator_s) == (0.5, 0.8)
