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


# Spoken when the interview finishes. The victim must hear that the request
# was received and rescuers are coming — that certainty is itself first aid.
CLOSING_FALLBACK = (
    "Thank you — you did really well. I've sent everything to the first "
    "responders. They've been notified, they know where you are, and they "
    "will reach you as fast as they can. Try to stay still and save your "
    "strength. Help is on the way."
)

# Spoken before the rover moves on from a survivor who never answered —
# they may be able to hear even if they can't speak.
NO_RESPONSE_CLOSING = (
    "If you can hear me: I have alerted the first responders. They know "
    "where you are, and they will reach you as fast as they can. "
    "Help is on the way."
)

_CLOSING_PROMPT = """You voice a rescue robot that has just finished interviewing one
survivor trapped under a collapsed building. Their answers have been sent on.
Write the ONE short spoken message the robot says before standing by.

It must:
- thank them warmly for answering — they may be in pain or scared.
- clearly confirm their request is received: the first responders have been
  notified, they know where the survivor is, and they will reach them as
  fast as they can.
- encourage them to hold on and save their strength.
- reassure without promises: "help is on the way" is fine; never give
  arrival times, never promise they will be okay.
- at most 45 words, everyday words, no medical jargon, never mention triage.
- the text is spoken by TTS: no lists, no emojis, no stage directions."""


def closing(history: list[dict], fields: dict) -> str:
    """The end-of-interview reassurance, adapted to the conversation. Falls
    back to the fixed message on any failure — the confirmation that help
    was notified must always be spoken."""
    try:
        convo = "\n".join(
            f"Robot: {h['question']}\nSurvivor: {h['answer']}" for h in history
        ) or "(no conversation recorded)"
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
                {"role": "system", "content": _CLOSING_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Conversation so far:\n{convo}\n\n"
                        f"What we learned: {json.dumps(known)}"
                    ),
                },
            ],
        )
        utterance = json.loads(response.choices[0].message.content)["utterance"].strip()
        return utterance or CLOSING_FALLBACK
    except Exception as exc:
        print(f"[interviewer] closing failed, using preset: {exc}")
        return CLOSING_FALLBACK


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
