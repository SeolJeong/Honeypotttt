"""Session -> LM input text (spec section 7), with ablation switches.

Pure Python, no torch/transformers dependency, so it can be shared by the
QLoRA model, the text baseline, the dashboard and the online scorer.

Output format (one block per interaction, oldest first):

    [STEP 1] [GAP=SHORT]
    ACTION: cat /etc/application.conf
    RESULT [not_found_with_hint]: No such file or directory. Try /opt/app/config instead.

Ablation switches map directly onto Experiment 6:
    include_responses=False   -> "commands only" (the central ablation)
    include_timing=False      -> "without timing"
    include_category=False    -> drop the [response_category] tag
    shuffle_steps=True        -> "commands without ordering"
"""
from __future__ import annotations

import json
import os
import random
import zlib
from dataclasses import asdict, dataclass

DATA_DIR = "data"
DS1 = os.path.join(DATA_DIR, "dataset1_sessions.json")

LABELS = ["human", "scripted", "AI-driven"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

# the same thresholds used to build dataset4 (scripts/fix_responses.py)
_GAP_BINS = [(2.0, "SHORT"), (6.0, "MEDIUM"), (15.0, "LONG")]

MAX_RESPONSE_CHARS = 400  # long outputs (ls -la, ps) are clipped, keep the head

INSTRUCTION = (
    "Classify the actor behind this SSH honeypot session as human, scripted "
    "or AI-driven, based on how each action reacts to the previous result."
)


@dataclass(frozen=True)
class FormatConfig:
    include_responses: bool = True
    include_timing: bool = True
    include_category: bool = True
    shuffle_steps: bool = False
    max_response_chars: int = MAX_RESPONSE_CHARS

    def tag(self) -> str:
        """Short name for logs and output folders."""
        parts = []
        if not self.include_responses:
            parts.append("noresp")
        if not self.include_timing:
            parts.append("notime")
        if not self.include_category:
            parts.append("nocat")
        if self.shuffle_steps:
            parts.append("shuffled")
        return "-".join(parts) or "full"

    def to_dict(self) -> dict:
        return asdict(self)


def gap_bin(seconds: float | None) -> str:
    if seconds is None:
        return "NONE"
    for threshold, label in _GAP_BINS:
        if seconds < threshold:
            return label
    return "VERY_LONG"


def _clip(text: str, n: int) -> str:
    text = text.strip()
    return text if len(text) <= n else text[:n] + " ...[clipped]"


def format_steps(steps: list[dict], cfg: FormatConfig = FormatConfig(),
                 seed: int = 0) -> str:
    """Render a list of interaction dicts (a full or partial session)."""
    steps = list(steps)
    if cfg.shuffle_steps:
        random.Random(seed).shuffle(steps)

    blocks = []
    for i, st in enumerate(steps, start=1):
        head = f"[STEP {i}]"
        if cfg.include_timing:
            head += f" [GAP={gap_bin(st.get('inter_command_gap_s'))}]"
        lines = [head, f"ACTION: {st['command']}"]
        if cfg.include_responses:
            tag = f" [{st['response_category']}]" if cfg.include_category else ""
            lines.append(f"RESULT{tag}: "
                         f"{_clip(st.get('system_response', ''), cfg.max_response_chars)}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def build_prompt(steps: list[dict], cfg: FormatConfig = FormatConfig(),
                 seed: int = 0) -> str:
    """Instruction + session trace. The classifier head reads the final token."""
    return (f"{INSTRUCTION}\n\n### SESSION\n{format_steps(steps, cfg, seed)}"
            f"\n\n### ACTOR:")


# ── dataset construction ────────────────────────────────────────────────

def load_sessions(path: str = DS1) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def prefix_lengths(n: int, cap: int | None = None) -> list[int]:
    """Prefix lengths used for online training. Always keeps 1-5 and the full
    session; long sessions are thinned to at most `cap` prefixes."""
    lengths = list(range(1, n + 1))
    if cap is None or n <= cap:
        return lengths
    head = [L for L in lengths if L <= 5]
    rest = lengths[5:-1]
    k = max(0, cap - len(head) - 1)
    picked = [rest[round(i * (len(rest) - 1) / max(1, k - 1))] for i in range(k)] if rest and k else []
    return sorted(set(head + picked + [n]))


def make_examples(sessions: list[dict], cfg: FormatConfig = FormatConfig(),
                  prefixes: bool = False, k: int | None = None,
                  max_prefixes: int | None = None) -> list[dict]:
    """Turn sessions into model examples.

    prefixes=True  -> one example per prefix length 1..n (online training:
                      the model must classify partial traces, spec section 10)
    k=int          -> only the first k steps of every session (early
                      attribution eval). Sessions shorter than k are used
                      in full, and `steps_seen` records what was really seen.
    """
    out = []
    for s in sessions:
        steps = s["interactions"]
        n = len(steps)
        lengths = prefix_lengths(n, max_prefixes) if prefixes else [min(k, n) if k else n]
        for L in lengths:
            out.append({
                "session_id": s["session_id"],
                "text": build_prompt(steps[:L], cfg, seed=zlib.crc32(s["session_id"].encode())),
                "label": LABEL2ID[s["actor_label"]],
                "actor_label": s["actor_label"],
                "evasion_strategy": s.get("evasion_strategy", "none"),
                "generator_family": s.get("generator_family", ""),
                "participant_id": s.get("participant_id"),
                "task": s.get("task", ""),
                "steps_seen": L,
                "total_steps": n,
            })
    return out


def grouped_val_split(train: list[dict], seed: int = 0,
                      ai_val_frac: float = 0.2) -> tuple[list[dict], list[dict]]:
    """Carve an early-stopping validation set out of the training split.

    Kept leakage-safe in the same way as the test split where the data allows:
      - human:    hold out one whole participant (the last one, e.g. P003)
      - scripted: hold out one whole script family
      - AI:       only two training families exist, so holding one out would
                  leave a single family to learn from; instead a stratified
                  ~20% of sessions (per family x evasion strategy) is held out.
    The test split is never touched.
    """
    rng = random.Random(seed)
    humans = sorted({s["participant_id"] for s in train if s["actor_label"] == "human"})
    scripts = sorted({s["generator_family"] for s in train if s["actor_label"] == "scripted"})
    val_participant = humans[-1] if len(humans) > 1 else None
    val_script = scripts[seed % len(scripts)] if len(scripts) > 1 else None

    ai_cells: dict[tuple, list[dict]] = {}
    for s in train:
        if s["actor_label"] == "AI-driven":
            ai_cells.setdefault((s["generator_family"], s["evasion_strategy"]), []).append(s)
    ai_val_ids = set()
    for cell in ai_cells.values():
        ids = [s["session_id"] for s in cell]
        rng.shuffle(ids)
        ai_val_ids.update(ids[:max(1, round(len(ids) * ai_val_frac))])

    def is_val(s):
        if s["actor_label"] == "human":
            return s["participant_id"] == val_participant
        if s["actor_label"] == "scripted":
            return s["generator_family"] == val_script
        return s["session_id"] in ai_val_ids

    tr = [s for s in train if not is_val(s)]
    va = [s for s in train if is_val(s)]
    return tr, va


def select_round(sessions: list[dict], round_: int) -> list[dict]:
    """Training-pool filter for the adversarial rounds (spec section 9).

    round 0 -> non-evasive training only (evasion_strategy == 'none')
    round 2 -> non-evasive + the training-split evasion strategies
    (round 1 is not a training regime: it is round-0 weights evaluated on
    evasive test sessions, which the evaluation report already breaks out)
    """
    if round_ == 0:
        return [s for s in sessions if s.get("evasion_strategy", "none") == "none"]
    return list(sessions)
