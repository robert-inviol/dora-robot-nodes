"""Turning what the Hailo pose post-process reports for one letterboxed model input into people in colour image pixels.

The post-process gives each person's box as shares (0..1) of the square model input, and each keypoint as shares of
that box. The model input is the colour image shrunk by a whole factor and centred on a square canvas.
"""

from dataclasses import dataclass

from dora_rig_messages.people import Keypoint, PersonPose, PersonPoses, PixelBox
from dora_rig_messages.status import TrackId


@dataclass(frozen=True)
class ModelInputFrame:
    """Where a letterboxed model input came from, read from the metadata of the camera node's `model_input`."""

    frame_id: int
    captured_camera_ns: int
    side: int
    source_width: int
    source_height: int
    shrink: int
    pad_x: int
    pad_y: int

    @classmethod
    def from_metadata(cls, metadata: dict) -> "ModelInputFrame":
        if metadata.get("encoding") != "rgb8" or metadata["width"] != metadata["height"]:
            raise ValueError(f"model input has to be square rgb8, received {metadata.get('encoding')} {metadata['width']}x{metadata['height']}")
        return cls(
            frame_id=metadata["frame_id"],
            captured_camera_ns=metadata["camera_ns"],
            side=metadata["width"],
            source_width=metadata["source_width"],
            source_height=metadata["source_height"],
            shrink=metadata["shrink"],
            pad_x=metadata["pad_x"],
            pad_y=metadata["pad_y"],
        )

    def to_source(self, x_share: float, y_share: float) -> tuple[float, float]:
        return (x_share * self.side - self.pad_x) * self.shrink, (y_share * self.side - self.pad_y) * self.shrink


@dataclass(frozen=True)
class HailoPerson:
    """One person as the post-process and tracker leave them: box in shares of the model input, keypoints in shares
    of the box."""

    track_id: int | None
    score: float
    xmin: float
    ymin: float
    width: float
    height: float
    keypoints: tuple[tuple[float, float, float], ...]


def person_poses(frame: ModelInputFrame, people: list[HailoPerson]) -> PersonPoses:
    def pose(person: HailoPerson) -> PersonPose:
        x0, y0 = frame.to_source(person.xmin, person.ymin)
        x1, y1 = frame.to_source(person.xmin + person.width, person.ymin + person.height)
        keypoints = tuple(
            Keypoint(*frame.to_source(person.xmin + x * person.width, person.ymin + y * person.height), score=score)
            for x, y, score in person.keypoints
        )
        track_id = None if person.track_id is None else TrackId(person.track_id)
        return PersonPose(track_id, person.score, PixelBox(x0, y0, x1 - x0, y1 - y0), keypoints)

    return PersonPoses(
        frame_id=frame.frame_id,
        captured_camera_ns=frame.captured_camera_ns,
        width=frame.source_width,
        height=frame.source_height,
        people=tuple(pose(person) for person in people),
    )
