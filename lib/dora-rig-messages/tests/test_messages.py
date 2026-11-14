import pyarrow as pa
import pytest

from dora_rig_messages.drive import STOPPED, DriveDemand, PowerCap
from dora_rig_messages.fixtures import EXAMPLES, read_fixture
from dora_rig_messages.operator import DriveMode, OperatorCommand
from dora_rig_messages.scan import SCAN_TYPE, Scan
from dora_rig_messages.status import DriverStatus, PilotStatus


@pytest.mark.parametrize("name", EXAMPLES)
def test_a_message_is_written_exactly_as_its_golden_fixture(name):
    assert EXAMPLES[name].to_arrow().equals(read_fixture(name))


@pytest.mark.parametrize("name", EXAMPLES)
def test_a_golden_fixture_is_read_back_as_its_message(name):
    example = EXAMPLES[name]

    assert type(example).from_arrow(read_fixture(name)) == example


def test_a_message_of_another_wire_type_is_rejected():
    with pytest.raises(ValueError, match="expected wire type"):
        DriveDemand.from_arrow(OperatorCommand.idle().to_arrow())


def test_an_output_carrying_more_than_one_message_is_rejected():
    two_rows = pa.concat_arrays([STOPPED.to_arrow(), STOPPED.to_arrow()])

    with pytest.raises(ValueError, match="expected one row"):
        DriveDemand.from_arrow(two_rows)


def test_a_scan_whose_bearings_and_ranges_are_out_of_step_is_rejected():
    out_of_step = pa.array(
        [{"spin_rev_per_s": 8.0, "bearing_deg": [0.0, 90.0], "range_m": [1.5]}], type=SCAN_TYPE
    )

    with pytest.raises(ValueError, match="2 bearings for 1 ranges"):
        Scan.from_arrow(out_of_step)


def test_a_wish_within_the_cap_is_scaled_by_the_cap():
    assert DriveDemand.limited(forward=1.0, turn=0.0, cap=PowerCap(20)) == DriveDemand(0.2, 0.0)


def test_a_wish_for_full_forward_and_full_turn_keeps_its_ratio_under_the_cap():
    assert DriveDemand.limited(forward=1.0, turn=1.0, cap=PowerCap(50)) == DriveDemand(0.25, 0.25)


@pytest.mark.parametrize("forward, turn", [(1.0, 1.0), (-1.0, 0.8), (0.3, -2.0), (0.0, 0.0)])
def test_a_limited_demand_never_asks_for_more_than_the_cap(forward, turn):
    demand = DriveDemand.limited(forward, turn, cap=PowerCap(40))

    assert abs(demand.forward) + abs(demand.turn) <= 0.4 + 1e-9


def test_a_demand_for_more_than_full_power_is_rejected():
    with pytest.raises(ValueError, match="more than full power"):
        DriveDemand(forward=0.8, turn=0.3)


def test_a_demand_for_exactly_full_power_is_accepted():
    assert DriveDemand(forward=0.75, turn=-0.25).forward == 0.75


@pytest.mark.parametrize("percent", [0, 101])
def test_a_power_cap_outside_1_to_100_percent_is_rejected(percent):
    with pytest.raises(ValueError, match="outside 1..100"):
        PowerCap(percent)


@pytest.mark.parametrize("throttle, steer", [(1.01, 0.0), (-1.01, 0.0), (0.0, 1.01), (0.0, -1.01)])
def test_a_stick_position_beyond_full_deflection_is_rejected(throttle, steer):
    with pytest.raises(ValueError, match="expected -1 to 1"):
        OperatorCommand(DriveMode.MANUAL, throttle, steer)


def test_an_unknown_mode_on_the_wire_is_rejected():
    unknown = pa.array(
        [{"mode": "autopilot", "throttle": 0.0, "steer": 0.0}],
        type=OperatorCommand.idle().to_arrow().type,
    )

    with pytest.raises(ValueError, match="autopilot"):
        OperatorCommand.from_arrow(unknown)


def test_an_idle_operator_asks_for_nothing():
    assert OperatorCommand.idle() == OperatorCommand(DriveMode.MANUAL, throttle=0.0, steer=0.0)


def test_status_messages_survive_the_trip_between_nodes():
    pilot = PilotStatus(DriveMode.MANUAL, locked_id=None)
    driver = DriverStatus(armed=False, left_pct=-50, right_pct=50)

    assert PilotStatus.from_arrow(pilot.to_arrow()) == pilot
    assert DriverStatus.from_arrow(driver.to_arrow()) == driver


def test_imu_samples_without_a_timestamp_each_are_rejected():
    import numpy as np

    from dora_rig_messages.perception import Imu

    with pytest.raises(ValueError, match="imu gyro has 1 timestamps"):
        Imu(0, 0, np.zeros(0, np.int64), np.zeros((0, 3), np.float32), np.zeros(1, np.int64), np.zeros((2, 3), np.float32))


def test_an_odom_pose_vector_of_the_wrong_length_is_rejected():
    from dataclasses import replace

    from dora_rig_messages.odometry import OdomPose

    pose = EXAMPLES["odom_pose"]
    assert isinstance(pose, OdomPose)
    with pytest.raises(ValueError, match="orientation_xyzw has 3 values, not 4"):
        replace(pose, orientation_xyzw=(0.0, 0.0, 1.0))
    wire = pose.to_arrow().to_pylist()[0] | {"position_m": [1.0, 2.0]}
    with pytest.raises(ValueError, match="position_m has 2 values, not 3"):
        OdomPose.from_arrow(pa.array([wire], type=pose.to_arrow().type))


@pytest.mark.parametrize("lone", ["weakest_direction", "weakest_sigma_m"])
def test_an_odom_pose_with_half_of_its_weakest_direction_is_rejected(lone):
    from dora_rig_messages.odometry import ODOM_POSE_TYPE, OdomPose

    half = EXAMPLES["odom_pose"].to_arrow().to_pylist()[0] | {"weakest_direction": None, "weakest_sigma_m": None}
    half[lone] = [0.0, 1.0, 0.0] if lone == "weakest_direction" else 0.125

    with pytest.raises(ValueError, match="one of weakest_direction and weakest_sigma_m without the other"):
        OdomPose.from_arrow(pa.array([half], type=ODOM_POSE_TYPE))


def test_an_odom_pose_missing_a_required_field_is_rejected():
    from dora_rig_messages.odometry import ODOM_POSE_TYPE, OdomPose

    unstamped = EXAMPLES["odom_pose"].to_arrow().to_pylist()[0] | {"camera_ns": None}

    with pytest.raises(ValueError, match="missing camera_ns"):
        OdomPose.from_arrow(pa.array([unstamped], type=ODOM_POSE_TYPE))
