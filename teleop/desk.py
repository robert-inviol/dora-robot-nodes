"""The state of the controls: which mode is selected and where the stick is."""

import threading

from follower.operator import DriveMode, OperatorCommand


class ControlDesk:
    """Moved by the web server thread and read by the dora loop, so every access takes the guard."""

    def __init__(self, stick_stale_after_s: float):
        self._stick_stale_after_s = stick_stale_after_s
        self._guard = threading.Lock()
        self._connected_pages = 0
        self._mode = DriveMode.MANUAL
        self._stick = OperatorCommand.idle()
        self._stick_moved_s = float("-inf")

    def page_connected(self) -> None:
        with self._guard:
            self._connected_pages += 1

    def page_disconnected(self) -> None:
        with self._guard:
            self._connected_pages -= 1
            if self._connected_pages == 0:
                # With no page open nobody can reach the stop button, so follow mode ends.
                self._mode = DriveMode.MANUAL

    def select_mode(self, mode: DriveMode) -> None:
        with self._guard:
            self._mode = mode
            self._stick = OperatorCommand.idle()

    def stop(self) -> None:
        self.select_mode(DriveMode.MANUAL)

    def move_stick(self, throttle: float, steer: float, now_s: float) -> None:
        """Touching the stick takes over from follow mode."""
        stick = OperatorCommand(DriveMode.MANUAL, throttle, steer)
        with self._guard:
            self._mode = DriveMode.MANUAL
            self._stick = stick
            self._stick_moved_s = now_s

    def command(self, now_s: float) -> OperatorCommand:
        with self._guard:
            if self._connected_pages == 0:
                return OperatorCommand.idle()
            if self._mode is DriveMode.FOLLOW:
                return OperatorCommand(DriveMode.FOLLOW, throttle=0.0, steer=0.0)
            if now_s - self._stick_moved_s > self._stick_stale_after_s:
                return OperatorCommand.idle()
            return self._stick
