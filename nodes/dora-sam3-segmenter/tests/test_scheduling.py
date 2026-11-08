from dora_sam3_segmenter.scheduling import ConceptRota


def test_newly_named_types_are_searched_for_on_the_next_keyframe():
    rota = ConceptRota(idle_per_keyframe=1)

    assert rota.add(["sofa", "rug"]) == ["sofa", "rug"]
    assert rota.for_next_keyframe() == ["sofa", "rug"]


def test_a_type_named_twice_is_added_once():
    rota = ConceptRota(idle_per_keyframe=1)
    rota.add(["sofa"])

    assert rota.add(["sofa", "rug"]) == ["rug"]


def test_found_types_are_always_searched_and_the_rest_take_turns():
    rota = ConceptRota(idle_per_keyframe=1)
    rota.add(["sofa", "rug", "lamp", "vase"])
    rota.finished(found={"sofa"})

    turns = [rota.for_next_keyframe() for _ in range(4)]

    assert turns == [["sofa", "rug"], ["sofa", "lamp"], ["sofa", "vase"], ["sofa", "rug"]]
