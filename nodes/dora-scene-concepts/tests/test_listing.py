from dora_scene_concepts.listing import ConceptLines


def test_a_line_is_released_only_once_it_is_finished():
    lines = ConceptLines()

    assert lines.feed("sofa\nred bl") == ["sofa"]
    assert lines.feed("anket\n") == ["red blanket"]


def test_repeats_bullets_and_blank_lines_are_dropped():
    lines = ConceptLines()

    assert lines.feed("- Sofa\n\n* sofa\n  Toy Dinosaur  \n") == ["sofa", "toy dinosaur"]


def test_the_last_line_without_a_newline_is_released_at_the_end():
    lines = ConceptLines()
    lines.feed("rug\nbirdcage")

    assert lines.finish() == ["birdcage"]
