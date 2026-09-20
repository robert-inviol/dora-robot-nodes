from pathlib import Path

import pytest

from dora_person_follower.config import ConfigError, load_config
from dora_person_follower.target import SelectionRule

SHIPPED_CONFIG = Path(__file__).parents[3] / "robots" / "tank" / "follower.toml"


def _config_with(tmp_path: Path, old: str, new: str) -> Path:
    shipped = SHIPPED_CONFIG.read_text()
    assert old in shipped
    path = tmp_path / "follower.toml"
    path.write_text(shipped.replace(old, new))
    return path


def test_the_shipped_config_is_capped_at_the_child_safe_power():
    assert load_config(SHIPPED_CONFIG).gains.max_power.percent == 20


def test_the_selection_rule_is_read(tmp_path):
    path = _config_with(tmp_path, '"largest"', '"most_central"')

    assert load_config(path).target.selection_rule is SelectionRule.MOST_CENTRAL


def test_an_unknown_selection_rule_is_rejected(tmp_path):
    path = _config_with(tmp_path, 'selection_rule = "largest"', 'selection_rule = "tallest"')

    with pytest.raises(ConfigError, match="tallest"):
        load_config(path)


@pytest.mark.parametrize("max_power_pct", [0, 101])
def test_a_follow_power_cap_outside_1_to_100_is_rejected(tmp_path, max_power_pct):
    path = _config_with(tmp_path, "max_power_pct = 20", f"max_power_pct = {max_power_pct}")

    with pytest.raises(ConfigError, match="pursuit.max_power_pct"):
        load_config(path)


@pytest.mark.parametrize("max_power_pct", [1, 100])
def test_a_power_cap_from_1_to_100_is_accepted(tmp_path, max_power_pct):
    path = _config_with(tmp_path, "max_power_pct = 20", f"max_power_pct = {max_power_pct}")

    assert load_config(path).gains.max_power.percent == max_power_pct


def test_a_missing_setting_is_named(tmp_path):
    path = _config_with(tmp_path, "turn_gain = 0.8", "")

    with pytest.raises(ConfigError, match="turn_gain"):
        load_config(path)


@pytest.mark.parametrize("max_power_pct", [0, 101])
def test_a_manual_power_cap_outside_1_to_100_is_rejected(tmp_path, max_power_pct):
    path = _config_with(tmp_path, "max_power_pct = 50", f"max_power_pct = {max_power_pct}")

    with pytest.raises(ConfigError, match="manual.max_power_pct"):
        load_config(path)


def test_both_stale_input_timeouts_are_read(tmp_path):
    path = _config_with(tmp_path, "stale_operator_after_s = 0.5", "stale_operator_after_s = 0.8")

    stale_after = load_config(path).stale_after

    assert (stale_after.detections_s, stale_after.operator_s) == (0.5, 0.8)


def test_a_forward_gain_of_zero_is_accepted_for_turning_on_the_spot(tmp_path):
    path = _config_with(tmp_path, "forward_gain = 0", "forward_gain = 0.0")

    assert load_config(path).gains.forward == 0


def test_a_negative_forward_gain_is_rejected(tmp_path):
    path = _config_with(tmp_path, "forward_gain = 0", "forward_gain = -0.5")

    with pytest.raises(ConfigError, match="pursuit.forward_gain"):
        load_config(path)
