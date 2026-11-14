import pyarrow as pa
import pytest

from dora_rig_messages.fixtures import read_fixture
from dora_rig_messages.navigation import (
    FIXTURE_EXAMPLES,
    SPACE_ANSWER_TYPE,
    GoalId,
    QuestionId,
    Reference,
    ReferenceKind,
    Revision,
    SpaceAnswer,
    SpaceQuestion,
)


@pytest.mark.parametrize("name", FIXTURE_EXAMPLES)
def test_a_navigation_message_round_trips_through_its_golden_fixture(name):
    example = FIXTURE_EXAMPLES[name]

    assert example.to_arrow().equals(read_fixture(name))
    assert type(example).from_arrow(read_fixture(name)) == example


def test_a_pose_reference_has_no_id_and_every_other_kind_has_one():
    with pytest.raises(ValueError, match="a pose reference takes no id"):
        Reference(ReferenceKind.POSE, 4, 1.0, 2.0)
    with pytest.raises(ValueError, match="a room reference needs an id"):
        Reference(ReferenceKind.ROOM, None, 1.0, 2.0)


def test_an_answer_without_references_or_goal_is_written_with_an_empty_list_and_a_null_goal():
    answer = SpaceAnswer(QuestionId(1), Revision(0), True, "claude-haiku-5-5", "Nothing like that here.", (), None)

    row = answer.to_arrow()

    assert row.type == SPACE_ANSWER_TYPE
    assert row[0].as_py()["references"] == []
    assert row[0].as_py()["goal_id"] is None
    assert SpaceAnswer.from_arrow(row) == answer


def test_a_question_is_not_read_as_an_answer():
    with pytest.raises(ValueError, match="expected wire type"):
        SpaceAnswer.from_arrow(SpaceQuestion(QuestionId(1), 0, "planner", "Hi?").to_arrow())


def test_an_unknown_reference_kind_is_rejected():
    row = pa.array(
        [
            {
                "question_id": 1,
                "revision": 0,
                "is_final": True,
                "model": "m",
                "text": "t",
                "references": [{"kind": "doorway", "id": 2, "x_m": 0.0, "y_m": 0.0}],
                "goal_id": GoalId(3),
            }
        ],
        type=SPACE_ANSWER_TYPE,
    )

    with pytest.raises(ValueError, match="'doorway' is not a valid ReferenceKind"):
        SpaceAnswer.from_arrow(row)
