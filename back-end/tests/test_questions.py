from app.questions import FIELD_ORDER, QUESTIONS, next_question


def unknown_fields() -> dict:
    return {f: None for f in FIELD_ORDER}


def test_first_responder_order():
    """Life safety first, then vitals, then injuries/mobility, then rescue intel."""
    fields = unknown_fields()
    asked: list[str] = []
    order = []
    while (q := next_question(fields, asked)) is not None:
        order.append(q.field)
        asked.append(q.id)
    assert order == FIELD_ORDER
    assert order[0] == "trapped"
    assert order[-2:] == ["people_in_building", "others_last_seen"]


def test_known_field_is_skipped():
    fields = unknown_fields()
    fields["trapped"] = True
    q = next_question(fields, [])
    assert q is not None and q.field == "breathing"


def test_false_and_zero_count_as_known():
    """Known means not-null — False and 0 are real values, not unknowns."""
    fields = unknown_fields()
    fields["trapped"] = False
    fields["breathing"] = False
    fields["respiratory_rate"] = 0
    q = next_question(fields, [])
    assert q is not None and q.field == "radial_pulse_present"


def test_ask_once_rule():
    """A field still unknown after its question was asked is never re-asked."""
    fields = unknown_fields()
    asked = [QUESTIONS["trapped"].id]
    q = next_question(fields, asked)
    assert q is not None and q.field == "breathing"


def test_answer_covering_later_field_skips_its_question():
    fields = unknown_fields()
    fields["trapped"] = True
    fields["breathing"] = True
    fields["respiratory_rate"] = 22
    asked = [QUESTIONS["trapped"].id]
    q = next_question(fields, asked)
    assert q is not None and q.field == "radial_pulse_present"


def test_others_last_seen_skipped_when_alone():
    """No point asking where the others are if the survivor was alone."""
    fields = unknown_fields()
    fields["people_in_building"] = 1
    asked = [QUESTIONS[f].id for f in FIELD_ORDER if f != "others_last_seen"]
    assert next_question(fields, asked) is None


def test_others_last_seen_asked_when_not_alone():
    fields = unknown_fields()
    fields["people_in_building"] = 4
    asked = [QUESTIONS[f].id for f in FIELD_ORDER if f != "others_last_seen"]
    q = next_question(fields, asked)
    assert q is not None and q.field == "others_last_seen"


def test_others_last_seen_asked_when_count_unknown():
    """Unknown count must not suppress the question — unknown is not 'alone'."""
    fields = unknown_fields()
    asked = [QUESTIONS[f].id for f in FIELD_ORDER if f != "others_last_seen"]
    q = next_question(fields, asked)
    assert q is not None and q.field == "others_last_seen"


def test_terminates_when_all_known():
    fields = {
        "trapped": True,
        "breathing": True,
        "respiratory_rate": 24,
        "radial_pulse_present": True,
        "obeys_commands": True,
        "can_walk": False,
        "injuries": "crushed ankle",
        "people_in_building": 4,
        "others_last_seen": "upstairs bedroom",
    }
    assert next_question(fields, []) is None


def test_terminates_when_all_asked_even_if_unknown():
    fields = unknown_fields()
    asked = [QUESTIONS[f].id for f in FIELD_ORDER]
    assert next_question(fields, asked) is None
