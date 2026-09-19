import pytest

from follower.drive import DriveCommand, TrackSpeed
from follower.operator import DriveMode, OperatorCommand, PilotStatus
from follower.people import TrackId


def test_an_operator_command_survives_the_trip_between_nodes():
    command = OperatorCommand(DriveMode.MANUAL, throttle=0.4, steer=-0.7)

    assert OperatorCommand.from_row(command.to_row()) == command


def test_an_idle_operator_asks_for_nothing():
    assert OperatorCommand.idle() == OperatorCommand(DriveMode.MANUAL, throttle=0.0, steer=0.0)


@pytest.mark.parametrize("throttle, steer", [(1.01, 0.0), (-1.01, 0.0), (0.0, 1.01), (0.0, -1.01)])
def test_a_stick_position_beyond_full_deflection_is_rejected(throttle, steer):
    with pytest.raises(ValueError, match="expected -1 to 1"):
        OperatorCommand(DriveMode.MANUAL, throttle, steer)


@pytest.mark.parametrize("deflection", [-1.0, 1.0])
def test_full_deflection_is_accepted(deflection):
    assert OperatorCommand(DriveMode.MANUAL, deflection, deflection).throttle == deflection


def test_an_unknown_mode_on_the_wire_is_rejected():
    with pytest.raises(ValueError, match="autopilot"):
        OperatorCommand.from_row({"mode": "autopilot", "throttle": 0.0, "steer": 0.0})


@pytest.mark.parametrize("locked_id", [None, TrackId(12)])
def test_a_pilot_status_survives_the_trip_between_nodes(locked_id):
    status = PilotStatus(
        mode=DriveMode.FOLLOW,
        locked_id=locked_id,
        drive=DriveCommand(TrackSpeed(14), TrackSpeed(-3)),
        armed=True,
    )

    assert PilotStatus.from_row(status.to_row()) == status
