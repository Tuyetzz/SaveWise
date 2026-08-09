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
higher = more urgent. Purely additive and explainable.

classify() emits its own reasoning trace as it evaluates: every condition
checked becomes a trace entry recording the field values it read and whether
it matched, in evaluation order. Rules after the first match are not
evaluated and therefore do not appear — the trace is what actually ran,
not a reconstruction."""

CORE_GATES = [
    "breathing",
    "respiratory_rate",
    "radial_pulse_present",
    "obeys_commands",
]

BASE = {"immediate": 300, "delayed": 200, "minor": 100}


def classify(fields: dict, status: str = "complete") -> dict:
    reasons: list[str] = []
    trace: list[dict] = []

    def record(
        phase: str,
        condition: str,
        evaluated: dict,
        result: bool,
        effect: str | None = None,
        points: int | None = None,
    ) -> bool:
        entry: dict = {
            "step": len(trace) + 1,
            "phase": phase,  # gate | rule | score
            "condition": condition,
            "evaluated": evaluated,
            "result": result,
        }
        if effect is not None:
            entry["effect"] = effect
        if points is not None:
            entry["points"] = points
        trace.append(entry)
        return result

    unknown_gates = [g for g in CORE_GATES if fields.get(g) is None]
    rr = fields.get("respiratory_rate")

    # START gates. All four are checked up front; the category rules below
    # consume the aggregate.
    gate_checks = [
        ("breathing", "breathing == false", fields.get("breathing") is False,
         "cannot breathe adequately"),
        ("respiratory_rate", "respiratory_rate >= 30", rr is not None and rr >= 30,
         f"respiratory rate {rr}/min — at or above 30"),
        ("radial_pulse_present", "radial_pulse_present == false",
         fields.get("radial_pulse_present") is False,
         "no radial pulse — poor perfusion"),
        ("obeys_commands", "obeys_commands == false",
         fields.get("obeys_commands") is False,
         "altered mental status"),
    ]
    failed_gates: list[str] = []
    failed_reasons: list[str] = []
    for name, condition, hit, reason in gate_checks:
        record("gate", condition, {name: fields.get(name)}, hit,
               effect=reason if hit else None)
        if hit:
            failed_gates.append(name)
            failed_reasons.append(reason)

    # Category rules, strict precedence — first match wins, later rules
    # are never evaluated.
    category: str | None = None
    rule_fired: str | None = None

    if record("rule", 'status == "no_response"', {"status": status},
              status == "no_response", effect="category = immediate"):
        category, rule_fired = "immediate", "no_response"
        reasons.append("no response to voice contact — possibly unconscious")
    if category is None and record(
        "rule", "can_walk == true", {"can_walk": fields.get("can_walk")},
        fields.get("can_walk") is True, effect="category = minor",
    ):
        category, rule_fired = "minor", "ambulatory"
        reasons.append("ambulatory — can walk out once a path is clear")
    if category is None and record(
        "rule", "any core gate failed", {"failed_gates": failed_gates},
        bool(failed_gates), effect="category = immediate",
    ):
        category, rule_fired = "immediate", "gate_failed"
        reasons.extend(failed_reasons)
    if category is None and record(
        "rule", "any core gate unknown", {"unknown_gates": unknown_gates},
        bool(unknown_gates),
        effect="category = immediate — unknown never assumed benign",
    ):
        category, rule_fired = "immediate", "unknown_escalated"
        reasons.extend(
            f"{g.replace('_', ' ')} unknown — escalated" for g in unknown_gates
        )
    if category is None:
        record("rule", "all gates known-good, not ambulatory", {}, True,
               effect="category = delayed")
        category, rule_fired = "delayed", "default_delayed"
        reasons.append("vitals within START limits, not ambulatory")

    score = BASE[category]
    record("score", f"base score — {category}", {"category": category}, True,
           points=BASE[category])
    if fields.get("trapped") is True:
        score += 25
        reasons.append("trapped or pinned — needs extrication")
        record("score", "trapped == true", {"trapped": True}, True, points=25,
               effect="needs extrication")
    elif fields.get("trapped") is None:
        score += 15
        reasons.append("entrapment unknown")
        record("score", "trapped unknown", {"trapped": None}, True, points=15,
               effect="entrapment unknown")
    else:
        record("score", "trapped == true", {"trapped": False}, False)
    # Confirmed failed gates outrank suspected (unknown) ones: a known-critical
    # casualty must never rank below an unknown one.
    if failed_gates:
        score += 12 * len(failed_gates)
        record("score", f"12 x {len(failed_gates)} confirmed failed gates",
               {"failed_gates": failed_gates}, True, points=12 * len(failed_gates))
    if unknown_gates:
        score += 8 * len(unknown_gates)
        record("score", f"8 x {len(unknown_gates)} unknown gates",
               {"unknown_gates": unknown_gates}, True, points=8 * len(unknown_gates))

    return {
        "category": category,
        "score": score,
        "reasons": reasons,
        "unknown_gates": unknown_gates,
        "rule_fired": rule_fired,
        "trace": trace,
    }
