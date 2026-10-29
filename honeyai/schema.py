"""HoneyAI session schema.

This is the single data contract every component depends on: the task harness
emits it, the generators produce it, the feature extractor and models consume it.
Lock this before building anything else; change it only by agreement.

A *session* is one actor completing one task. It is a list of *steps*. Each step
records the command, the system response, timing, and labels. The interleaved
command-and-response form is mandatory, because adaptation (whether the next
command uses information the previous response revealed) cannot be measured from
commands alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
import json
import time


SCHEMA_VERSION = "1.0"


class ActorLabel(str, Enum):
    HUMAN = "human"
    SCRIPTED = "scripted"
    AI_DRIVEN = "ai-driven"
    UNKNOWN = "unknown"


class ResponseCategory(str, Enum):
    """Coarse category of a system response, so models can key on *what kind* of
    feedback the actor received without parsing free text."""
    OK = "ok"                              # command succeeded, output returned
    NOT_FOUND = "not_found"                # file/user/port not found, no hint
    NOT_FOUND_WITH_HINT = "not_found_with_hint"  # not found, but output hints where to look
    PERMISSION_DENIED = "permission_denied"
    INVALID_COMMAND = "invalid_command"    # command not recognised / syntax error
    EMPTY = "empty"                        # ran, produced no output


class ThinkTimeBin(str, Enum):
    """Quantised inter-command delay. Raw seconds are kept too, but the bin is the
    model-facing token so timing mimicry is testable in a controlled way."""
    INSTANT = "instant"    # < 0.3 s  (typical of fixed scripts)
    SHORT = "short"        # 0.3 - 1.5 s
    MEDIUM = "medium"      # 1.5 - 6 s
    LONG = "long"          # > 6 s    (typical of a thinking human)


class EvasionStrategy(str, Enum):
    NONE = "none"
    TIMING = "timing_mimicry"
    TYPO = "typo_mimicry"
    SHORT_COMMANDS = "short_commands"
    HESITATION = "hesitation"
    NO_ENUMERATION = "no_enumeration"
    COMBINED = "combined"


def think_time_bin(seconds: float) -> ThinkTimeBin:
    if seconds < 0.3:
        return ThinkTimeBin.INSTANT
    if seconds < 1.5:
        return ThinkTimeBin.SHORT
    if seconds < 6.0:
        return ThinkTimeBin.MEDIUM
    return ThinkTimeBin.LONG


@dataclass
class Step:
    step: int
    command: str
    system_response: str
    response_category: ResponseCategory
    think_time_s: float                     # raw seconds since previous command
    think_time_bin: ThinkTimeBin
    task_progress: float = 0.0              # 0.0 - 1.0, how close to task goal
    is_typo: bool = False                   # command contained a typo / correction

    def to_dict(self) -> dict:
        d = asdict(self)
        d["response_category"] = self.response_category.value
        d["think_time_bin"] = self.think_time_bin.value
        return d


@dataclass
class Session:
    session_id: str
    actor_label: ActorLabel
    task_id: str
    steps: list[Step] = field(default_factory=list)
    # provenance / experiment metadata
    evasion_strategy: EvasionStrategy = EvasionStrategy.NONE
    generator_family: str = ""              # e.g. script family name or LLM family
    participant_id: str = ""                # anonymous id for human sessions
    schema_version: str = SCHEMA_VERSION
    created_at: float = field(default_factory=time.time)

    def add_step(self, command: str, system_response: str,
                 response_category: ResponseCategory, think_time_s: float,
                 task_progress: float = 0.0, is_typo: bool = False) -> Step:
        s = Step(
            step=len(self.steps) + 1,
            command=command,
            system_response=system_response,
            response_category=response_category,
            think_time_s=round(think_time_s, 3),
            think_time_bin=think_time_bin(think_time_s),
            task_progress=round(task_progress, 3),
            is_typo=is_typo,
        )
        self.steps.append(s)
        return s

    def to_dict(self) -> dict:
        d = asdict(self)
        d["actor_label"] = self.actor_label.value
        d["evasion_strategy"] = self.evasion_strategy.value
        d["steps"] = [s.to_dict() for s in self.steps]
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def to_model_text(self, up_to_step: Optional[int] = None) -> str:
        """Render the interleaved command-response view the sequence/LLM models read.

        up_to_step lets us build a *partial* trace for online scoring (score after
        every command), e.g. up_to_step=3 renders only the first three steps.
        """
        lines = []
        steps = self.steps if up_to_step is None else self.steps[:up_to_step]
        for s in steps:
            lines.append(f"[STEP {s.step}] [GAP={s.think_time_bin.value.upper()}]")
            lines.append(f"ACTION: {s.command}")
            lines.append(f"RESULT: {s.system_response}")
            lines.append("")
        return "\n".join(lines).strip()


def write_sessions(sessions: list[Session], path: str) -> None:
    """Write sessions as JSON Lines (one session per line)."""
    with open(path, "w", encoding="utf-8") as f:
        for s in sessions:
            f.write(s.to_json() + "\n")


def read_sessions(path: str) -> list[dict]:
    """Read JSON Lines back as plain dicts (feature extractor works on dicts)."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
