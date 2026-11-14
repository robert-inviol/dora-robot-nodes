"""Navigation goals an agent sends the robot, and the robot's reports on how each goal is going.

A goal names its target by id wherever it can, so the robot resolves where the target is when it acts, not where it
was when the agent decided. Poses are in the map frame: x and y in metres, yaw in radians counter-clockwise from x.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import NewType

import pyarrow as pa

from .wire import one_row, only_row

GoalId = NewType("GoalId", int)
ObjectId = NewType("ObjectId", int)
PlaceId = NewType("PlaceId", int)
MapVersion = NewType("MapVersion", int)


class GoalKind(Enum):
    # Drive to a map pose.
    POSE = "pose"
    # Drive to standoff_m from an object.
    OBJECT = "object"
    # Drive to a place node.
    PLACE = "place"
    # Hand the people tracker's track to the follower.
    FOLLOW_TRACK = "follow_track"
    # Turn on the spot to face an object or a map point.
    LOOK_AT = "look_at"
    # Stop and drop whatever goal is running.
    STOP = "stop"


class TargetForm(Enum):
    ID = "id"
    POSE = "pose"
    ID_OR_POSE = "id or pose"
    NOTHING = "nothing"


# The target each kind takes: an object, place or track id by kind; for look_at, an object id or a map point.
TARGET_FORM = {
    GoalKind.POSE: TargetForm.POSE,
    GoalKind.OBJECT: TargetForm.ID,
    GoalKind.PLACE: TargetForm.ID,
    GoalKind.FOLLOW_TRACK: TargetForm.ID,
    GoalKind.LOOK_AT: TargetForm.ID_OR_POSE,
    GoalKind.STOP: TargetForm.NOTHING,
}

MAP_POSE_TYPE = pa.struct([("x_m", pa.float32()), ("y_m", pa.float32()), ("yaw_rad", pa.float32())])

NAV_GOAL_TYPE = pa.struct(
    [
        ("goal_id", pa.uint64()),
        ("kind", pa.string()),
        # An object, place or track id, by kind; null when the target is a pose or there is none.
        ("target_id", pa.uint64()),
        # Null unless the target is a pose.
        ("target_pose", MAP_POSE_TYPE),
        ("standoff_m", pa.float32()),
        # The robot caps this again with its own limits.
        ("max_speed_m_s", pa.float32()),
        # The robot drops the goal when this passes.
        ("deadline_unix_ns", pa.int64()),
        # The map the agent read the target from.
        ("map_version", pa.uint64()),
    ]
)


@dataclass(frozen=True)
class MapPose:
    x_m: float
    y_m: float
    yaw_rad: float


def _finite_not_negative(name: str, value: float) -> None:
    if not (math.isfinite(value) and value >= 0.0):
        raise ValueError(f"{name} is {value}, expected a finite value of at least 0")


@dataclass(frozen=True)
class NavGoal:
    goal_id: GoalId
    kind: GoalKind
    target_id: int | None
    target_pose: MapPose | None
    standoff_m: float
    max_speed_m_s: float
    deadline_unix_ns: int
    map_version: MapVersion

    def __post_init__(self) -> None:
        form = TARGET_FORM[self.kind]
        has_id, has_pose = self.target_id is not None, self.target_pose is not None
        allowed = {
            TargetForm.ID: has_id and not has_pose,
            TargetForm.POSE: has_pose and not has_id,
            TargetForm.ID_OR_POSE: has_id != has_pose,
            TargetForm.NOTHING: not has_id and not has_pose,
        }[form]
        if not allowed:
            raise ValueError(f"a {self.kind.value} goal takes {form.value} as its target")
        _finite_not_negative("standoff_m", self.standoff_m)
        _finite_not_negative("max_speed_m_s", self.max_speed_m_s)

    def to_arrow(self) -> pa.Array:
        pose = self.target_pose
        return one_row(
            {
                "goal_id": self.goal_id,
                "kind": self.kind.value,
                "target_id": self.target_id,
                "target_pose": None if pose is None else {"x_m": pose.x_m, "y_m": pose.y_m, "yaw_rad": pose.yaw_rad},
                "standoff_m": self.standoff_m,
                "max_speed_m_s": self.max_speed_m_s,
                "deadline_unix_ns": self.deadline_unix_ns,
                "map_version": self.map_version,
            },
            NAV_GOAL_TYPE,
        )

    @classmethod
    def from_arrow(cls, array: pa.Array) -> "NavGoal":
        row = only_row(array, NAV_GOAL_TYPE)
        pose = row["target_pose"]
        return cls(
            goal_id=GoalId(row["goal_id"]),
            kind=GoalKind(row["kind"]),
            target_id=row["target_id"],
            target_pose=None if pose is None else MapPose(pose["x_m"], pose["y_m"], pose["yaw_rad"]),
            standoff_m=row["standoff_m"],
            max_speed_m_s=row["max_speed_m_s"],
            deadline_unix_ns=row["deadline_unix_ns"],
            map_version=MapVersion(row["map_version"]),
        )


class GoalState(Enum):
    ACCEPTED = "accepted"
    ACTIVE = "active"
    BLOCKED = "blocked"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


# The states that need a reason; the others carry none.
STATES_WITH_REASON = frozenset({GoalState.BLOCKED, GoalState.FAILED})


class GoalReason(Enum):
    UNKNOWN_TARGET = "unknown_target"
    NO_PATH = "no_path"
    TARGET_LOST = "target_lost"
    PERSON_IN_PATH = "person_in_path"
    STALE_MAP = "stale_map"
    OPERATOR_OVERRIDE = "operator_override"
    DEADLINE_PASSED = "deadline_passed"
    PREEMPTED = "preempted"


GOAL_STATUS_TYPE = pa.struct(
    [
        ("goal_id", pa.uint64()),
        ("state", pa.string()),
        # Null unless the state is blocked or failed.
        ("reason", pa.string()),
        ("updated_unix_ns", pa.int64()),
    ]
)


@dataclass(frozen=True)
class GoalStatus:
    goal_id: GoalId
    state: GoalState
    reason: GoalReason | None
    updated_unix_ns: int

    def __post_init__(self) -> None:
        if (self.state in STATES_WITH_REASON) != (self.reason is not None):
            needs = "needs" if self.state in STATES_WITH_REASON else "takes no"
            raise ValueError(f"a {self.state.value} goal status {needs} reason")

    def to_arrow(self) -> pa.Array:
        return one_row(
            {
                "goal_id": self.goal_id,
                "state": self.state.value,
                "reason": None if self.reason is None else self.reason.value,
                "updated_unix_ns": self.updated_unix_ns,
            },
            GOAL_STATUS_TYPE,
        )

    @classmethod
    def from_arrow(cls, array: pa.Array) -> "GoalStatus":
        row = only_row(array, GOAL_STATUS_TYPE)
        reason = row["reason"]
        return cls(
            goal_id=GoalId(row["goal_id"]),
            state=GoalState(row["state"]),
            reason=None if reason is None else GoalReason(reason),
            updated_unix_ns=row["updated_unix_ns"],
        )


QuestionId = NewType("QuestionId", int)
Revision = NewType("Revision", int)
RoomId = NewType("RoomId", int)

SPACE_QUESTION_TYPE = pa.struct(
    [
        ("question_id", pa.uint64()),
        ("asked_unix_ns", pa.int64()),
        # Who asked: the name of the agent or person, for the logs.
        ("asker", pa.string()),
        ("text", pa.string()),
    ]
)


@dataclass(frozen=True)
class SpaceQuestion:
    """A question about the space for the spatial agent; its answers come back as space_answer messages."""

    question_id: QuestionId
    asked_unix_ns: int
    asker: str
    text: str

    def to_arrow(self) -> pa.Array:
        return one_row(
            {"question_id": self.question_id, "asked_unix_ns": self.asked_unix_ns, "asker": self.asker, "text": self.text},
            SPACE_QUESTION_TYPE,
        )

    @classmethod
    def from_arrow(cls, array: pa.Array) -> "SpaceQuestion":
        row = only_row(array, SPACE_QUESTION_TYPE)
        return cls(QuestionId(row["question_id"]), row["asked_unix_ns"], row["asker"], row["text"])


class ReferenceKind(Enum):
    OBJECT = "object"
    PLACE = "place"
    ROOM = "room"
    TRACK = "track"
    # A raw map point, with no id.
    POSE = "pose"


REFERENCE_TYPE = pa.struct(
    [
        ("kind", pa.string()),
        # The object, place, room or track id; null for a pose.
        ("id", pa.uint64()),
        ("x_m", pa.float32()),
        ("y_m", pa.float32()),
    ]
)

SPACE_ANSWER_TYPE = pa.struct(
    [
        ("question_id", pa.uint64()),
        # 0 for the first answer; each later message for the same question counts up.
        ("revision", pa.uint32()),
        # True on the last message for the question.
        ("is_final", pa.bool_()),
        # The model whose answer this is.
        ("model", pa.string()),
        ("text", pa.string()),
        # What the answer is about, where it is in the map.
        ("references", pa.list_(pa.field("item", REFERENCE_TYPE, nullable=False))),
        # The nav goal the robot is on because of this answer; null when there is none.
        ("goal_id", pa.uint64()),
    ]
)


@dataclass(frozen=True)
class Reference:
    """Something an answer is about, and where it is in the map frame."""

    kind: ReferenceKind
    id: int | None
    x_m: float
    y_m: float

    def __post_init__(self) -> None:
        if (self.kind == ReferenceKind.POSE) != (self.id is None):
            needs = "takes no" if self.kind == ReferenceKind.POSE else "needs an"
            raise ValueError(f"a {self.kind.value} reference {needs} id")


@dataclass(frozen=True)
class SpaceAnswer:
    question_id: QuestionId
    revision: Revision
    is_final: bool
    model: str
    text: str
    references: tuple[Reference, ...]
    goal_id: GoalId | None

    def to_arrow(self) -> pa.Array:
        return one_row(
            {
                "question_id": self.question_id,
                "revision": self.revision,
                "is_final": self.is_final,
                "model": self.model,
                "text": self.text,
                "references": [
                    {"kind": r.kind.value, "id": r.id, "x_m": r.x_m, "y_m": r.y_m} for r in self.references
                ],
                "goal_id": self.goal_id,
            },
            SPACE_ANSWER_TYPE,
        )

    @classmethod
    def from_arrow(cls, array: pa.Array) -> "SpaceAnswer":
        row = only_row(array, SPACE_ANSWER_TYPE)
        goal = row["goal_id"]
        return cls(
            question_id=QuestionId(row["question_id"]),
            revision=Revision(row["revision"]),
            is_final=row["is_final"],
            model=row["model"],
            text=row["text"],
            references=tuple(
                Reference(ReferenceKind(r["kind"]), r["id"], r["x_m"], r["y_m"]) for r in row["references"]
            ),
            goal_id=None if goal is None else GoalId(goal),
        )


# Golden examples of this module's messages, merged into fixtures.EXAMPLES. Every float is exact in float32.
FIXTURE_EXAMPLES: dict[str, object] = {
    "space_question": SpaceQuestion(
        question_id=QuestionId(31),
        asked_unix_ns=1_762_560_000_250_000_000,
        asker="planner",
        text="Where is my blue mug?",
    ),
    "space_answer_first": SpaceAnswer(
        question_id=QuestionId(31),
        revision=Revision(0),
        is_final=False,
        model="claude-haiku-5-5",
        text="The blue mug o51 is on the kitchen table o42; I am heading to it.",
        references=(
            Reference(ReferenceKind.OBJECT, 51, 2.25, 1.5),
            Reference(ReferenceKind.OBJECT, 42, 2.0, 1.5),
        ),
        goal_id=GoalId(17),
    ),
    "space_answer_final": SpaceAnswer(
        question_id=QuestionId(31),
        revision=Revision(2),
        is_final=True,
        model="claude-opus-5-5",
        text="The blue mug is not here; the one on the table is white. Your blue mug was last seen by the sink.",
        references=(Reference(ReferenceKind.ROOM, 3, 1.5, 2.75), Reference(ReferenceKind.POSE, None, 0.75, 2.5)),
        goal_id=None,
    ),
}
