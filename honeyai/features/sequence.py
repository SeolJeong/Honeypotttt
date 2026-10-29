"""Sequence encoding for the recurrent model (Model 2).

Model 1 collapses a session to aggregate numbers. Model 2 instead sees the ordered
events, so it can learn *sequential* structure (e.g. a hint response immediately
followed by a command that uses the hint). This is what isolates whether sequence
information, not model scale, drives any improvement.

We encode each step as a small token sequence:
  <program token> <response-category token> <think-time-bin token>
plus two scalar features (command length, task progress) fed alongside the embedding.

A command vocabulary is built from the training data; unseen programs map to <unk>.
This keeps the model light enough to train on CPU for the PoC, and it consumes the
same interleaved view the spec mandates.
"""
from __future__ import annotations

from collections import Counter

PAD, UNK = "<pad>", "<unk>"
RESP_CATS = ["ok", "not_found", "not_found_with_hint", "permission_denied",
             "invalid_command", "empty"]
TIME_BINS = ["instant", "short", "medium", "long"]


def _program(cmd: str) -> str:
    return cmd.strip().split()[0] if cmd.strip() else PAD


class SequenceVocab:
    def __init__(self, programs: list[str]):
        self.prog2i = {PAD: 0, UNK: 1}
        for p in programs:
            if p not in self.prog2i:
                self.prog2i[p] = len(self.prog2i)
        self.resp2i = {c: i for i, c in enumerate(RESP_CATS)}
        self.time2i = {b: i for i, b in enumerate(TIME_BINS)}

    @property
    def n_programs(self) -> int:
        return len(self.prog2i)

    @classmethod
    def build(cls, sessions: list[dict], min_count: int = 1) -> "SequenceVocab":
        c = Counter()
        for s in sessions:
            for step in s["steps"]:
                c[_program(step["command"])] += 1
        progs = [p for p, n in c.items() if n >= min_count]
        return cls(progs)

    def encode(self, session: dict, max_len: int = 40):
        """Return per-step feature rows: [prog_id, resp_id, time_id, cmd_len_norm, progress].
        Truncated/padded to max_len. Returns (rows, length)."""
        rows = []
        for step in session["steps"][:max_len]:
            prog = _program(step["command"])
            rows.append([
                self.prog2i.get(prog, 1),
                self.resp2i.get(step["response_category"], 0),
                self.time2i.get(step["think_time_bin"], 0),
                min(len(step["command"]) / 40.0, 1.0),
                step["task_progress"],
            ])
        length = len(rows)
        while len(rows) < max_len:
            rows.append([0, 0, 0, 0.0, 0.0])
        return rows, length
