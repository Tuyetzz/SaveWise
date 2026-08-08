from app.questions import FIELD_ORDER
from app.triage import classify


def fields(**known) -> dict:
    f = {name: None for name in FIELD_ORDER}
    f.update(known)
    return f


GOOD_GATES = dict(
    breathing=True, respiratory_rate=18, radial_pulse_present=True, obeys_commands=True
)


def test_ambulatory_is_minor():
    result = classify(fields(can_walk=True, **GOOD_GATES, trapped=False))
    assert result["category"] == "minor"


def test_all_gates_good_not_ambulatory_is_delayed():
    result = classify(fields(can_walk=False, **GOOD_GATES, trapped=False))
    assert result["category"] == "delayed"


def test_fast_breathing_is_immediate():
    result = classify(
        fields(can_walk=False, **{**GOOD_GATES, "respiratory_rate": 34}, trapped=False)
    )
    assert result["category"] == "immediate"


def test_no_pulse_is_immediate():
    result = classify(
        fields(can_walk=False, **{**GOOD_GATES, "radial_pulse_present": False})
    )
    assert result["category"] == "immediate"


def test_unknown_gate_escalates_to_immediate():
    """Unknown never clears a gate — safety constraint, not a preference."""
    partial = fields(can_walk=False, breathing=True, trapped=False)
    result = classify(partial)
    assert result["category"] == "immediate"
    assert "respiratory_rate" in result["unknown_gates"]


def test_no_response_is_immediate():
    result = classify(fields(), status="no_response")
    assert result["category"] == "immediate"


def test_trapped_ranks_above_untrapped_same_category():
    trapped = classify(fields(can_walk=False, **GOOD_GATES, trapped=True))
    free = classify(fields(can_walk=False, **GOOD_GATES, trapped=False))
    assert trapped["category"] == free["category"] == "delayed"
    assert trapped["score"] > free["score"]


def test_immediate_always_outranks_delayed_and_minor():
    immediate = classify(fields(can_walk=False, **{**GOOD_GATES, "breathing": False}))
    delayed = classify(fields(can_walk=False, **GOOD_GATES, trapped=True))
    minor = classify(fields(can_walk=True, trapped=True))
    assert immediate["score"] > delayed["score"] > minor["score"]
