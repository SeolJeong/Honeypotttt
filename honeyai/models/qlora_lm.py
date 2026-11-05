"""Part 7: QLoRA fine-tuned LM utilities (Model 3).

Shared constants and helpers for Colab training + local integration.
The actual training runs on a Colab GPU notebook.

Usage:
    Training: paste the Colab cells (see project handoff)
    Results:  qlora_lm.load_predictions("data/qlora_predictions.json")
"""
from __future__ import annotations

import json

LABEL_NAMES = ["human", "scripted", "AI-driven"]
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT = (
    "You are a security analyst classifying SSH honeypot sessions. "
    "Given an interleaved command-response transcript, classify the "
    "actor as human, scripted, or AI-driven. "
    'Respond with JSON only: {"class": "human|scripted|AI-driven", '
    '"confidence": 0.xx}'
)


def format_messages(interleaved_text: str, label: str | None = None) -> list[dict]:
    """Build chat messages for the LLM. Include label for training."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": interleaved_text},
    ]
    if label is not None:
        target = json.dumps({"class": label, "confidence": 0.95})
        messages.append({"role": "assistant", "content": target})
    return messages


def parse_prediction(text: str) -> tuple[str, float]:
    """Extract class and confidence from model output text."""
    text = text.strip()
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
            cls = data.get("class", "")
            conf = float(data.get("confidence", 0.0))
            if cls in LABEL_NAMES:
                return cls, conf
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    # fallback: substring match
    for label in LABEL_NAMES:
        if label.lower() in text.lower():
            return label, 0.5
    return "unknown", 0.0


def load_predictions(path: str) -> list[dict]:
    """Load saved predictions JSON (from Colab output)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)