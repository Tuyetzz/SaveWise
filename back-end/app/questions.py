"""Deterministic question policy. Pure function — no LLM, no I/O, no randomness.

The LLM never chooses the next question. The interview is a rescue rover
speaking directly to a survivor buried under a collapsed building, so every
question is phrased for self-report. Order follows first-responder practice:
life safety first (entrapment, breathing), then vitals, then injuries and
mobility, then rescue intelligence for planning (how many people, where)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    id: str
    field: str
    text: str


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
    ),
    "breathing": Question(
        "q_breathing",
        "breathing",
        "Are you able to breathe okay right now?",
    ),
    "respiratory_rate": Question(
        "q_respiratory_rate",
        "respiratory_rate",
        "Try counting your breaths for fifteen seconds, then tell me the number.",
    ),
    "radial_pulse_present": Question(
        "q_radial_pulse",
        "radial_pulse_present",
        "If one of your hands is free, press two fingers on the inside of your "
        "wrist. Can you feel your pulse?",
    ),
    "obeys_commands": Question(
        "q_obeys_commands",
        "obeys_commands",
        "Do you feel fully alert, or do you feel confused or drowsy?",
    ),
    "can_walk": Question(
        "q_can_walk",
        "can_walk",
        "If rescuers cleared a path for you, do you think you could stand up "
        "and walk out on your own?",
    ),
    "injuries": Question(
        "q_injuries",
        "injuries",
        "Are you injured? Tell me where it hurts the most.",
    ),
    "people_in_building": Question(
        "q_people_in_building",
        "people_in_building",
        "How many people were inside the building when it collapsed, "
        "including you?",
    ),
    "others_last_seen": Question(
        "q_others_last_seen",
        "others_last_seen",
        "Where did you last see the other people who were with you?",
    ),
}

QUESTIONS_BY_ID: dict[str, Question] = {q.id: q for q in QUESTIONS.values()}


def next_question(fields: dict, asked: list[str]) -> Question | None:
    """First unknown field in ask order whose question has not been asked.

    Each question is asked at most once; a field still unknown after its
    question stays unknown (unknown escalates in Phase 2, so this is safe).
    Returns None when every field is either known or already asked."""
    for field in FIELD_ORDER:
        # No point asking where the others are if the survivor was alone.
        if field == "others_last_seen":
            people = fields.get("people_in_building")
            if people is not None and people <= 1:
                continue
        question = QUESTIONS[field]
        if fields.get(field) is None and question.id not in asked:
            return question
    return None
