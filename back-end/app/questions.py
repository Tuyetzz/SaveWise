"""Deterministic question policy. Pure function — no LLM, no I/O, no randomness.

The LLM never chooses the next goal. The interview is a rescue rover speaking
directly to a survivor buried under a collapsed building. Order follows
first-responder practice: life safety first (entrapment, breathing), then
vitals, then injuries and mobility, then rescue intelligence for planning
(how many people, where).

Each Question is a GOAL, not a script: interviewer.phrase() asks the LLM to
word it for the actual survivor (in pain, confused, shaken), and `text` is the
preset fallback phrasing if generation fails. Triage-critical fields that stay
unknown after one ask get exactly one circle-back at the end of the interview;
after that they stay unknown (and unknown escalates, so giving up is safe)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    id: str
    field: str
    text: str  # preset fallback phrasing
    goal: str  # what this turn must find out — the LLM phrases around this


# Ask order. The first six keep their START field names for Phase 2 triage;
# the last three feed rescue planning.
FIELD_ORDER = [
    "trapped",
    "breathing",
    "respiratory_rate",
    "radial_pulse_present",
    "obeys_commands",
    "can_walk",
    "injuries",
    "people_in_building",
    "others_last_seen",
]

QUESTIONS: dict[str, Question] = {
    "trapped": Question(
        "q_trapped",
        "trapped",
        "Hello? Can you hear me? I'm with the rescue team, and help is on "
        "the way. Stay with me — first, are you trapped or pinned by "
        "anything, or can you move your body?",
        "whether they are physically trapped or pinned down, or free to move",
    ),
    "breathing": Question(
        "q_breathing",
        "breathing",
        "Are you able to breathe okay right now?",
        "whether they can breathe adequately right now",
    ),
    "respiratory_rate": Question(
        "q_respiratory_rate",
        "respiratory_rate",
        "Try counting your breaths for fifteen seconds, then tell me the number.",
        "how fast they are breathing — ask them to count breaths for fifteen "
        "seconds and say the number",
    ),
    "radial_pulse_present": Question(
        "q_radial_pulse",
        "radial_pulse_present",
        "If one of your hands is free, press two fingers on the inside of your "
        "wrist. Can you feel your pulse?",
        "whether they can feel a pulse at the inside of their wrist "
        "(two fingers, if a hand is free)",
    ),
    "obeys_commands": Question(
        "q_obeys_commands",
        "obeys_commands",
        "Do you feel fully alert, or do you feel confused or drowsy?",
        "whether they feel fully alert, versus confused, drowsy, or fading",
    ),
    "can_walk": Question(
        "q_can_walk",
        "can_walk",
        "If rescuers cleared a path for you, do you think you could stand up "
        "and walk out on your own?",
        "whether they could stand up and walk out on their own if a path "
        "were cleared",
    ),
    "injuries": Question(
        "q_injuries",
        "injuries",
        "Are you injured? Tell me where it hurts the most.",
        "where they are hurt, in their own words",
    ),
    "people_in_building": Question(
        "q_people_in_building",
        "people_in_building",
        "How many people were inside the building when it collapsed, "
        "including you?",
        "how many people were inside the building when it collapsed, "
        "including them",
    ),
    "others_last_seen": Question(
        "q_others_last_seen",
        "others_last_seen",
        "Where did you last see the other people who were with you?",
        "where they last saw the other people who were inside",
    ),
}

QUESTIONS_BY_ID: dict[str, Question] = {q.id: q for q in QUESTIONS.values()}

# Fields worth one gentle second try if the first answer didn't settle them.
# The last three are rescue intel — useful, but not worth pressing a hurt,
# confused person twice for.
CRITICAL_FIELDS = [
    "trapped",
    "breathing",
    "respiratory_rate",
    "radial_pulse_present",
    "obeys_commands",
    "can_walk",
]


def next_question(fields: dict, asked: list[str]) -> Question | None:
    """Next goal: first unknown field in ask order not yet asked; once every
    goal has had a first ask, circle back once to triage-critical fields that
    are still unknown (the survivor may have been confused or in pain the
    first time). `asked` may contain an id twice — count, not membership.

    Never more than two asks per field; a field still unknown after that
    stays unknown (unknown escalates in Phase 2, so giving up is safe).
    Returns None when there is nothing left worth asking."""

    def skip(field: str) -> bool:
        # No point asking where the others are if the survivor was alone.
        if field == "others_last_seen":
            people = fields.get("people_in_building")
            return people is not None and people <= 1
        return False

    for field in FIELD_ORDER:
        question = QUESTIONS[field]
        if not skip(field) and fields.get(field) is None and asked.count(question.id) == 0:
            return question
    for field in CRITICAL_FIELDS:
        question = QUESTIONS[field]
        if fields.get(field) is None and asked.count(question.id) == 1:
            return question
    return None
