import os

os.environ.setdefault("OPEN_AI_KEY", "test-key")  # client built at import

import json
from types import SimpleNamespace

from app import interviewer
from app.questions import QUESTIONS


def _response(payload: dict):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
    )


def test_uses_generated_utterance(monkeypatch):
    monkeypatch.setattr(
        interviewer._client.chat.completions,
        "create",
        lambda **kwargs: _response(
            {"utterance": "You're doing well. Can you breathe okay right now?"}
        ),
    )
    text = interviewer.phrase(QUESTIONS["breathing"], [], {})
    assert text == "You're doing well. Can you breathe okay right now?"


def test_falls_back_to_preset_on_api_error(monkeypatch):
    """The interview must never block on the LLM — preset phrasing ships."""

    def boom(**kwargs):
        raise RuntimeError("api down")

    monkeypatch.setattr(interviewer._client.chat.completions, "create", boom)
    text = interviewer.phrase(QUESTIONS["breathing"], [], {})
    assert text == QUESTIONS["breathing"].text


def test_falls_back_to_preset_on_empty_utterance(monkeypatch):
    monkeypatch.setattr(
        interviewer._client.chat.completions,
        "create",
        lambda **kwargs: _response({"utterance": "   "}),
    )
    text = interviewer.phrase(QUESTIONS["breathing"], [], {})
    assert text == QUESTIONS["breathing"].text


def test_conversation_goal_and_attempt_reach_the_model(monkeypatch):
    captured: dict = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _response({"utterance": "ok"})

    monkeypatch.setattr(interviewer._client.chat.completions, "create", fake_create)
    interviewer.phrase(
        QUESTIONS["radial_pulse_present"],
        [{"question": "Are you hurt?", "answer": "my leg... I can't think straight"}],
        {"trapped": True},
        attempt=2,
    )
    user_msg = captured["messages"][1]["content"]
    assert "my leg... I can't think straight" in user_msg
    assert "pulse" in user_msg
    assert '"trapped": true' in user_msg
    assert "Attempt number for this question: 2" in user_msg
