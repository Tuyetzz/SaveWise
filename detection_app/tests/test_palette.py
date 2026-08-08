from rescue_vision.palette import PERSON_COLOURS_HEX, colour_for, hex_to_bgr


def test_the_first_three_slots_are_the_validated_all_pairs_set():
    """3 colours pass all-pairs separation; 6 hard-fail the normal-vision
    floor. The PRD assumes 1-3 people, so these three carry the real case."""
    assert PERSON_COLOURS_HEX[:3] == ["#2a78d6", "#eb6834", "#1baf7a"]


def test_each_person_gets_a_distinct_colour():
    assert len({colour_for(i) for i in range(1, 9)}) == 8


def test_colour_is_stable_for_a_given_sighting():
    assert colour_for(2) == colour_for(2)


def test_colour_follows_the_sighting_id_not_call_order():
    """A person leaving must not repaint the survivors."""
    first = colour_for(3)
    colour_for(1)
    colour_for(2)
    assert colour_for(3) == first


def test_slots_wrap_after_eight_people():
    assert colour_for(9) == colour_for(1)


def test_hex_converts_to_bgr_for_opencv():
    assert hex_to_bgr("#2a78d6") == (214, 120, 42)
