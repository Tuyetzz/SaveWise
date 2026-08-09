"""LLM question phrasing. The deterministic policy (questions.py) decides WHAT
to ask next — the goal of the turn; this module decides HOW to say it, adapting
to a survivor who may be in pain, confused, panicking, or rambling.

The LLM never picks the next goal, never extracts fields, and never touches
triage. If generation fails for any reason, the preset phrasing ships instead —
the interview must never block on a language model."""

import json
import os

from openai import OpenAI

from app.questions import Question

_client = OpenAI(api_key=os.environ["OPEN_AI_KEY"])

_SCHEMA = {
    "type": "object",
    "properties": {"utterance": {"type": "string"}},
    "required": ["utterance"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = """You voice a rescue robot talking with one survivor trapped under a
collapsed building. A deterministic system has already chosen the goal of this
turn; your only job is to phrase the next thing said out loud.

The survivor may be in pain, confused, shaken, or barely holding on. Be patient
with them, every time, no matter how they answer.

Write ONE short spoken utterance that:
- first briefly acknowledges what they just said, warmly and specifically —
  especially pain, fear, or confusion. Never dismiss it, never argue, never
  say "calm down".
- then asks about the goal: one plain question, at most 30 words total,
  everyday words, no medical jargon.
- if nothing has been said yet, this is first contact: say who you are and
  that help is on the way before asking.
- if the attempt number is above 1, they did not answer or were confused —
  do not repeat yourself verbatim. Reassure first, then ask a simpler,
  smaller version of the same thing.
- if this goal was asked about earlier and is being revisited, acknowledge
  that ("I know I asked before...") and make it easier to answer.
- if their last answer was rambling or off-topic, keep whatever they gave
  you ("I've noted that") and gently steer back — no scolding.
- reassure without promises: "help is on the way" is fine; never give arrival
  times, never promise they will be okay, never give medical instructions
  beyond the goal question itself.
- never mention triage, categories, priority, or scores.
- the text is spoken by TTS: no lists, no emojis, no stage directions."""


def phrase(
    question: Question,
    history: list[dict],
    fields: dict,
    attempt: int = 1,
) -> str:
    """One utterance for this turn's goal. history is the conversation so far:
    [{"question": spoken_text, "answer": transcript}, ...] in order. Falls
    back to the preset phrasing on any failure or empty generation."""
    try:
        convo = "\n".join(
            f"Robot: {h['question']}\nSurvivor: {h['answer']}" for h in history
        ) or "(first contact — nothing said yet)"
        known = {k: v for k, v in fields.items() if v is not None}
        response = _client.chat.completions.create(
            model="gpt-5.6-luna",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "utterance",
                    "strict": True,
                    "schema": _SCHEMA,
                },
            },
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Conversation so far:\n{convo}\n\n"
                        f"Already known: {json.dumps(known)}\n\n"
                        f"Goal of this turn: find out {question.goal}\n"
                        f"Reference phrasing (adapt to the person, don't parrot): "
                        f"{question.text}\n"
                        f"Attempt number for this question: {attempt}"
                    ),
                },
            ],
        )
        utterance = json.loads(response.choices[0].message.content)["utterance"].strip()
        return utterance or question.text
    except Exception as exc:
        print(f"[interviewer] phrasing failed, using preset: {exc}")
        return question.text
