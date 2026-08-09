from app.questions import FIELD_ORDER, QUESTIONS, next_question


def unknown_fields() -> dict:
    return {f: None for f in FIELD_ORDER}


def test_first_responder_order():
    """Life safety first, then vitals, then injuries/mobility, then rescue
    intel — then one circle-back over still-unknown critical fields."""
    from app.questions import CRITICAL_FIELDS

    fields = unknown_fields()
    asked: list[str] = []
    order = []
    while (q := next_question(fields, asked)) is not None:
        order.append(q.field)
        asked.append(q.id)
    assert order == FIELD_ORDER + CRITICAL_FIELDS
    assert order[0] == "trapped"
    assert order[: len(FIELD_ORDER)][-2:] == ["people_in_building", "others_last_seen"]


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


def test_unasked_goals_come_before_any_retry():
    """A field still unknown after its ask waits until every goal has had a
    first ask — the survivor is never pressed twice in a row."""
    fields = unknown_fields()
    asked = [QUESTIONS["trapped"].id]
    q = next_question(fields, asked)
    assert q is not None and q.field == "breathing"


def test_critical_field_circled_back_once():
    """A triage-critical field still unknown after the first pass gets one
    gentle retry at the end, then stays unknown (unknown escalates, so
    giving up is safe)."""
    fields = unknown_fields()
    asked = [QUESTIONS[f].id for f in FIELD_ORDER]  # first pass done
    q = next_question(fields, asked)
    assert q is not None and q.field == "trapped"
    asked.append(q.id)
    q = next_question(fields, asked)
    assert q is not None and q.field == "breathing"


def test_answered_field_not_circled_back():
    fields = unknown_fields()
    fields["trapped"] = False
    asked = [QUESTIONS[f].id for f in FIELD_ORDER]
    q = next_question(fields, asked)
    assert q is not None and q.field == "breathing"


def test_intel_fields_never_re_asked():
    """Rescue intel (injuries, people, last seen) is not worth pressing a
    hurt, confused person twice for."""
    fields = unknown_fields()
    asked = [QUESTIONS[f].id for f in FIELD_ORDER]
    asked += [QUESTIONS[f].id for f in FIELD_ORDER if f not in
              ("injuries", "people_in_building", "others_last_seen")]
    assert next_question(fields, asked) is None


def test_never_more_than_two_asks():
    fields = unknown_fields()
    asked = [QUESTIONS[f].id for f in FIELD_ORDER] * 2
    assert next_question(fields, asked) is None


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
    fields = {
        "trapped": True,
        "breathing": True,
        "respiratory_rate": 24,
        "radial_pulse_present": True,
        "obeys_commands": True,
        "can_walk": False,
        "injuries": "arm",
        "people_in_building": 1,
        "others_last_seen": None,
    }
    assert next_question(fields, []) is None


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
    """After the first pass plus one circle-back over critical fields, the
    interview ends even if everything is still unknown."""
    from app.questions import CRITICAL_FIELDS

    fields = unknown_fields()
    asked = [QUESTIONS[f].id for f in FIELD_ORDER]
    asked += [QUESTIONS[f].id for f in CRITICAL_FIELDS]
    assert next_question(fields, asked) is None
