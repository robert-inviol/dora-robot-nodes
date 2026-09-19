"""People parsed from the detection rows the perception bridge publishes."""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from messages.status import TrackId

PERSON_CLASS_NAME = "person"
# The bridge's track_id for a detection that hailotracker has not numbered yet.
UNTRACKED = -1


@dataclass(frozen=True)
class Person:
    """A tracked person in normalised frame coordinates: 0..1, origin at the top left."""

    track_id: TrackId
    centre_x: float
    frame_fill: float
    area: float


def people_in(rows: Sequence[Mapping[str, Any]], min_confidence: float) -> list[Person]:
    return [
        Person(
            track_id=TrackId(row["track_id"]),
            centre_x=row["x"] + row["w"] / 2,
            frame_fill=row["h"],
            area=row["w"] * row["h"],
        )
        for row in rows
        if row["class_name"] == PERSON_CLASS_NAME
        and row["confidence"] >= min_confidence
        and row["track_id"] != UNTRACKED
    ]
