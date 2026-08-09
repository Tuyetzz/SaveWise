"""LLM field extraction. The LLM extracts clinical fields only — it never
assigns a triage category and never chooses the next question. Unsure -> null;
unknown escalates later, so null must never default to a benign value."""

import json
import os

from openai import OpenAI

from app.questions import FIELD_ORDER, QUESTIONS_BY_ID

_client = OpenAI(api_key=os.environ["OPEN_AI_KEY"])

_SCHEMA = {
    "type": "object",
    "properties": {
        "trapped": {"type": ["boolean", "null"]},
        "breathing": {"type": ["boolean", "null"]},
        "respiratory_rate": {"type": ["integer", "null"]},
        "radial_pulse_present": {"type": ["boolean", "null"]},
        "obeys_commands": {"type": ["boolean", "null"]},
        "can_walk": {"type": ["boolean", "null"]},
        "injuries": {"type": ["string", "null"]},
        "people_in_building": {"type": ["integer", "null"]},
        "others_last_seen": {"type": ["string", "null"]},
    },
    "required": FIELD_ORDER,
    "additionalProperties": False,
}

_SYSTEM_PROMPT = """You extract triage and rescue-planning fields from one transcribed
utterance of a survivor trapped under a collapsed building, answering a rescue
robot. Return only the JSON fields.

Rules:
- Set a field only when the utterance clearly states it. If unsure, use null.
- Never guess, never infer beyond what was said. null means unknown. The fact
  that the survivor is speaking does not by itself establish any field.
- breathing: true if they say they can breathe adequately, false if they say
  they cannot or are struggling.
- respiratory_rate is breaths per minute as an integer. If they give a count
  over fifteen seconds, multiply by four.
- obeys_commands: true if they clearly state they are alert and oriented,
  false if they say they are confused, drowsy, or fading.
- can_walk: whether they say they could stand and walk out if a path allowed.
- injuries: a short summary of stated injuries, in their words.
- people_in_building: total people inside when it collapsed, including the
  survivor.
- others_last_seen: where the other occupants were last seen, as stated.
- Never assign, suggest, or mention a triage category.
- The utterance may cover fields other than the one asked about; extract those
  too."""


def extract(
    transcript: str, question_id: str, prior: dict, question_text: str | None = None
) -> dict:
    """One extraction call, merged over prior. A null in the response never
    overwrites a known prior value. On any failure, return prior unchanged —
    the interview continues and the field stays unknown.

    question_text is the phrasing actually spoken (questions are LLM-worded
    per survivor); the preset text is the fallback context."""
    question = QUESTIONS_BY_ID.get(question_id)
    if question_text is None:
        question_text = question.text if question else question_id
    try:
        # gpt-5.6-luna only supports the default temperature (rejects 0);
        # determinism relies on the strict JSON schema instead.
        response = _client.chat.completions.create(
            model="gpt-5.6-luna",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "triage_fields",
                    "strict": True,
                    "schema": _SCHEMA,
                },
            },
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Question asked: {question_text}\n"
                        f"Transcribed answer: {transcript}\n"
                        f"Fields already known: "
                        f"{json.dumps({k: v for k, v in prior.items() if v is not None})}"
                    ),
                },
            ],
        )
        extracted = json.loads(response.choices[0].message.content)
        merged = dict(prior)
        for field in FIELD_ORDER:
            value = extracted.get(field)
            if value is not None:
                merged[field] = value
        return merged
    except Exception as exc:
        print(f"[extraction] failed, keeping prior fields: {exc}")
        return prior
