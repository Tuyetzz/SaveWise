"""Deterministic START triage classification. Pure function — no LLM, no I/O.

Adapted START for self-report interviews:
- ambulatory -> minor (green)
- any failed gate (breathing, resp rate >= 30, no radial pulse, altered
  mental status) -> immediate (red)
- all gates known-good, not ambulatory -> delayed (yellow)

Unknown never defaults to benign: a core gate still unknown escalates the case
to immediate rather than clearing it. A survivor who never answered at all
(status no_response) is treated as possibly unconscious -> immediate.

The score orders cases within and across categories for the dashboard;
higher = more urgent. Purely additive and explainable."""

CORE_GATES = [
    "breathing",
    "respiratory_rate",
    "radial_pulse_present",
    "obeys_commands",
]

BASE = {"immediate": 300, "delayed": 200, "minor": 100}


def classify(fields: dict, status: str = "complete") -> dict:
    reasons: list[str] = []
    unknown_gates = [g for g in CORE_GATES if fields.get(g) is None]
    rr = fields.get("respiratory_rate")

    failed = []
    if fields.get("breathing") is False:
        failed.append("cannot breathe adequately")
    if rr is not None and rr >= 30:
        failed.append(f"respiratory rate {rr}/min — at or above 30")
    if fields.get("radial_pulse_present") is False:
        failed.append("no radial pulse — poor perfusion")
    if fields.get("obeys_commands") is False:
        failed.append("altered mental status")

    if status == "no_response":
        category = "immediate"
        reasons.append("no response to voice contact — possibly unconscious")
    elif fields.get("can_walk") is True:
        category = "minor"
        reasons.append("ambulatory — can walk out once a path is clear")
    elif failed:
        category = "immediate"
        reasons.extend(failed)
    elif unknown_gates:
        category = "immediate"
        reasons.extend(
            f"{g.replace('_', ' ')} unknown — escalated" for g in unknown_gates
        )
    else:
        category = "delayed"
        reasons.append("vitals within START limits, not ambulatory")

    score = BASE[category]
    if fields.get("trapped") is True:
        score += 25
        reasons.append("trapped or pinned — needs extrication")
    elif fields.get("trapped") is None:
        score += 15
        reasons.append("entrapment unknown")
    # Confirmed failed gates outrank suspected (unknown) ones: a known-critical
    # casualty must never rank below an unknown one.
    score += 12 * len(failed)
    score += 8 * len(unknown_gates)

    return {
        "category": category,
        "score": score,
        "reasons": reasons,
        "unknown_gates": unknown_gates,
    }
