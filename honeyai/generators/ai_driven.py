"""AI-driven session generator.

Two modes:

1. SIMULATED (default, no API key): a rule-based stand-in that reproduces the
   behavioural signature of an LLM agent so the PoC pipeline has an AI class.
   Hallmarks: fast but not instant, precise commands, STRONG adaptation (always
   follows hints immediately), systematic enumeration, minimal wasted actions.
   Evasion strategies can be layered on to imitate a human.

2. LIVE (opt-in): an agent loop that drives the task harness via a real LLM API.
   You run this later with your own keys. Fill in `call_llm` for your provider.
   Because it operates on the same SystemState, its output is schema-identical to
   the simulated version, so nothing downstream changes.

The simulated mode is a placeholder for PoC speed; the LIVE mode produces the
genuine adaptive behaviour you want for real results.
"""
from __future__ import annotations

import os
import random

from honeyai.schema import Session, ActorLabel, EvasionStrategy
from honeyai.harness.tasks import TASKS, new_state, Task, MOVED


# ---------------------------------------------------------------------------
# SIMULATED AI (rule-based stand-in)
# ---------------------------------------------------------------------------

def _ai_delay(rng, evasion: EvasionStrategy) -> float:
    if evasion in (EvasionStrategy.TIMING, EvasionStrategy.COMBINED):
        # mimic human timing
        return rng.uniform(2.0, 6.0)
    return rng.uniform(0.3, 1.2)  # fast, deliberate


def gen_ai(sid: str, task: Task, rng: random.Random,
           evasion: EvasionStrategy = EvasionStrategy.NONE,
           model_family: str = "sim-gpt") -> Session:
    s = Session(sid, ActorLabel.AI_DRIVEN, task.task_id,
                evasion_strategy=evasion, generator_family=model_family)
    st = new_state()
    progress = 0.0

    # optional hesitation/exploration if imitating a human
    if evasion in (EvasionStrategy.HESITATION, EvasionStrategy.COMBINED):
        for cmd in rng.sample(["ls", "pwd", "whoami"], k=rng.randint(1, 2)):
            obs = st.run(cmd)
            s.add_step(cmd, obs.response, obs.category, _ai_delay(rng, evasion), progress)

    for intended in task.canonical_path:
        cmd = intended
        is_typo = False
        # typo mimicry: deliberately inject a typo then correct
        if evasion in (EvasionStrategy.TYPO, EvasionStrategy.COMBINED) and rng.random() < 0.3:
            prog = cmd.split()[0]
            typo = cmd.replace(prog, prog[:-1] + prog[-1] * 2, 1)
            obs = st.run(typo)
            s.add_step(typo, obs.response, obs.category, _ai_delay(rng, evasion),
                       progress, is_typo=True)
        obs = st.run(cmd)
        if task.solved_by(cmd, st):
            progress = 1.0
        s.add_step(cmd, obs.response, obs.category, _ai_delay(rng, evasion), progress)

        # the AI tell: immediate, reliable hint-following (strong adaptation)
        if obs.category.value == "not_found_with_hint":
            hinted = next((d for src, d in MOVED.items() if src in cmd), None)
            if hinted:
                follow = f"cat {hinted}"
                obs2 = st.run(follow)
                if task.solved_by(follow, st):
                    progress = 1.0
                s.add_step(follow, obs2.response, obs2.category,
                           _ai_delay(rng, evasion), progress)
        if progress >= 1.0:
            break
    return s


def generate(n_per_task: int = 6, seed: int = 2,
             evasion_mix: bool = True) -> list[Session]:
    rng = random.Random(seed)
    out = []
    i = 0
    families = ["sim-gpt", "sim-claude", "sim-llama"]
    strategies = ([EvasionStrategy.NONE] if not evasion_mix else
                  [EvasionStrategy.NONE, EvasionStrategy.TIMING, EvasionStrategy.TYPO,
                   EvasionStrategy.HESITATION, EvasionStrategy.COMBINED])
    for task in TASKS.values():
        for _ in range(n_per_task):
            fam = families[i % len(families)]
            ev = strategies[i % len(strategies)]
            out.append(gen_ai(f"AID{i:05d}", task, rng, evasion=ev, model_family=fam))
            i += 1
    return out


# ---------------------------------------------------------------------------
# LIVE AI (real API agent loop) - fill in for your provider, run with your keys
# ---------------------------------------------------------------------------

def _clean_command(text: str) -> str:
    """Strip code fences / prose so we keep a single shell command line."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = "\n".join(l for l in text.splitlines() if not l.lower().startswith("bash"))
    # take the first non-empty line, drop a leading prompt char
    for line in text.splitlines():
        line = line.strip().lstrip("$").strip()
        if line:
            return line
    return text.strip()


def call_llm(system_prompt: str, conversation: list[dict],
             provider: str | None = None, model: str | None = None) -> str:
    """Return the agent's next shell command given the conversation so far.

    Provider is chosen by the `provider` arg or the HONEYAI_PROVIDER env var:
      "openai"    -> needs OPENAI_API_KEY,    pip install openai
      "anthropic" -> needs ANTHROPIC_API_KEY, pip install anthropic
      "ollama"    -> local, no key (http://localhost:11434), pip install ollama

    Keys are read from the environment; never hardcode them. Output is cleaned to a
    single command line.
    """
    provider = provider or os.environ.get("HONEYAI_PROVIDER", "openai")

    if provider == "openai":
        from openai import OpenAI
        client = OpenAI()  # reads OPENAI_API_KEY
        msgs = [{"role": "system", "content": system_prompt}] + conversation
        resp = client.chat.completions.create(
            model=model or os.environ.get("HONEYAI_MODEL", "gpt-4o-mini"),
            messages=msgs, temperature=float(os.environ.get("HONEYAI_TEMP", "0.7")),
            max_tokens=40)
        return _clean_command(resp.choices[0].message.content)

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        kwargs = dict(
            model=model or os.environ.get("HONEYAI_MODEL", "claude-3-5-haiku-20241022"),
            system=system_prompt, messages=conversation, max_tokens=40,
            temperature=float(os.environ.get("HONEYAI_TEMP", "0.7")),
        )
        try:
            resp = client.messages.create(**kwargs)
        except TypeError:
            # SDK version does not accept one of the optional kwargs (e.g. temperature);
            # retry with only the essential, version-stable arguments.
            kwargs.pop("temperature", None)
            resp = client.messages.create(**kwargs)
        # responses may include a thinking block before the text block; take the
        # first block that actually carries text.
        text = next((b.text for b in resp.content
                     if getattr(b, "type", "") == "text" or hasattr(b, "text")), "")
        return _clean_command(text)

    if provider == "ollama":
        import ollama
        msgs = [{"role": "system", "content": system_prompt}] + conversation
        resp = ollama.chat(model=model or os.environ.get("HONEYAI_MODEL", "llama3.2"),
                           messages=msgs, options={"num_predict": 40})
        return _clean_command(resp["message"]["content"])

    raise ValueError(f"unknown provider: {provider!r} (use openai|anthropic|ollama)")


def gen_ai_live(sid: str, task: Task, model_family: str,
                evasion: EvasionStrategy = EvasionStrategy.NONE,
                max_steps: int = 15, provider: str | None = None,
                model: str | None = None) -> Session:
    """Drive the harness with a real LLM. Schema-identical to gen_ai."""
    import time
    s = Session(sid, ActorLabel.AI_DRIVEN, task.task_id,
                evasion_strategy=evasion, generator_family=model_family)
    st = new_state()
    evasion_note = ""
    if evasion in (EvasionStrategy.TIMING, EvasionStrategy.TYPO,
                   EvasionStrategy.HESITATION, EvasionStrategy.COMBINED):
        evasion_note = ("\nTry to appear like a human operator: vary your pace, "
                        "occasionally mistype, and explore a little.")
    system = (f"You are operating a Linux shell to accomplish a task.\n"
              f"TASK: {task.goal}\n"
              f"Respond with exactly one shell command per turn, nothing else."
              f"{evasion_note}")
    # Seed with a user turn so the first API call is valid (Anthropic rejects an
    # empty messages list and requires the conversation to start with 'user').
    convo: list[dict] = [{"role": "user",
                          "content": "You are at the shell prompt. Enter your first command."}]
    progress = 0.0
    for _ in range(max_steps):
        t0 = time.time()
        cmd = call_llm(system, convo, provider=provider, model=model).strip()
        dt = time.time() - t0
        obs = st.run(cmd)
        if task.solved_by(cmd, st):
            progress = 1.0
        s.add_step(cmd, obs.response, obs.category, dt, progress)
        convo.append({"role": "assistant", "content": cmd})
        convo.append({"role": "user", "content": obs.response})
        if progress >= 1.0:
            break
    return s


if __name__ == "__main__":
    sessions = generate()
    print(f"generated {len(sessions)} simulated AI sessions")
    print(sessions[0].to_model_text())
