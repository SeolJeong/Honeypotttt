"""HoneyAI dashboard: per-command attack-attribution replay.

Replays a honeypot session command-by-command and shows, after every command:
  - live class probabilities (human / scripted / AI-driven)
  - the running prediction
  - the adaptive-threat score
  - the interleaved command-response transcript so far

This is the "real-time" view for the demo. It uses deterministic replay of recorded
sessions, which is both honest (per-command scoring is genuine) and demo-safe (it
cannot fail live on stage). Point it at live Cowrie-derived sessions later and the
same code streams them.

Run:  python -m honeyai.dashboard.app
Then open the printed local URL.
"""
from __future__ import annotations

import os
import gradio as gr

from honeyai.schema import read_sessions
from honeyai.scoring import train_default_scorer
from honeyai.attack_mapping import map_session, map_command

DATA = os.path.join("data", "sessions", "poc_dataset.jsonl")

SESSIONS = read_sessions(DATA)
SCORER = train_default_scorer(SESSIONS)
# index a readable label per session for the dropdown
CHOICES = {
    f"{s['session_id']}  [{s['actor_label']}]  {s.get('generator_family','')}"
    f"{'/'+s['evasion_strategy'] if s.get('evasion_strategy') not in (None,'none') else ''}": i
    for i, s in enumerate(SESSIONS)
}


def _bar(label, value):
    pct = int(round(value * 100))
    colour = {"human": "#2f9e44", "scripted": "#5c7cfa", "ai-driven": "#e8590c"}[label]
    return (f"<div style='margin:4px 0;font:13px sans-serif'>"
            f"<span style='display:inline-block;width:90px'>{label}</span>"
            f"<span style='display:inline-block;width:240px;background:#eee;border-radius:4px;vertical-align:middle'>"
            f"<span style='display:inline-block;height:14px;width:{pct*2.4}px;background:{colour};border-radius:4px'></span>"
            f"</span> <b>{pct}%</b></div>")


def replay(choice, k):
    i = CHOICES[choice]
    s = SESSIONS[i]
    n = len(s["steps"])
    k = max(1, min(int(k), n))
    r = SCORER.score_step(s, k)

    bars = "".join(_bar(l, r["probs"][l]) for l in ["human", "scripted", "ai-driven"])
    threat_pct = int(r["threat"] * 100)
    threat_colour = "#e03131" if threat_pct >= 60 else "#f08c00" if threat_pct >= 30 else "#2f9e44"

    # WHO
    header = (f"<div style='font:14px sans-serif'>"
              f"<b>WHO &mdash; attacker type after step {k}/{n}:</b> "
              f"<span style='color:#1f2933'>{r['prediction'].upper()}</span><br>"
              f"<b>Adaptive-threat score:</b> "
              f"<span style='color:{threat_colour};font-size:16px'><b>{threat_pct}</b></span>/100</div>")

    # WHAT - ATT&CK techniques seen so far (first k steps)
    prefix = {**s, "steps": s["steps"][:k]}
    atk = map_session(prefix)
    if atk["techniques"]:
        chips = "".join(
            f"<span style='display:inline-block;background:#f1f3f5;border:1px solid #dee2e6;"
            f"border-radius:4px;padding:2px 6px;margin:2px;font:12px sans-serif'>"
            f"{t}</span>" for t in atk["techniques"])
        tactics = ", ".join(atk["tactics"].keys())
        what = (f"<div style='font:13px sans-serif;margin-top:10px'>"
                f"<b>WHAT &mdash; ATT&CK techniques observed:</b><br>{chips}"
                f"<br><span style='color:#495057'>Tactics: {tactics}</span></div>")
    else:
        what = ("<div style='font:13px sans-serif;margin-top:10px;color:#868e96'>"
                "<b>WHAT &mdash; ATT&CK:</b> nothing classified yet</div>")

    # transcript with per-command technique tag
    transcript = []
    for j in range(k):
        step = s["steps"][j]
        m = map_command(step["command"])
        tag = f"   [{m[0]}]" if m else ""
        transcript.append(f"$ {step['command']}{tag}")
        transcript.append(f"  {step['system_response'][:90]}")
    transcript_txt = "\n".join(transcript)

    truth = (f"<div style='font:12px sans-serif;color:#868e96;margin-top:10px'>"
             f"ground truth: {s['actor_label']} &middot; family: {s.get('generator_family','')} "
             f"&middot; evasion: {s.get('evasion_strategy','none')}</div>")

    return header + "<br>" + bars + what + truth, transcript_txt


def on_select(choice):
    i = CHOICES[choice]
    n = len(SESSIONS[i]["steps"])
    return gr.update(maximum=n, value=1)


with gr.Blocks(title="HoneyAI") as demo:
    gr.Markdown("# HoneyAI SIEM dashboard\n"
                "**WHO** is attacking (human / script / AI) + **WHAT** they are doing "
                "(MITRE ATT&CK) + **threat** score, updating after every command. "
                "Pick a session and step through it.")
    with gr.Row():
        with gr.Column(scale=1):
            dd = gr.Dropdown(choices=list(CHOICES.keys()), label="Session",
                             value=list(CHOICES.keys())[0])
            sl = gr.Slider(1, 10, value=1, step=1, label="Commands observed")
        with gr.Column(scale=2):
            out = gr.HTML()
            tr = gr.Code(label="Transcript so far", language="shell")
    dd.change(on_select, dd, sl).then(replay, [dd, sl], [out, tr])
    sl.change(replay, [dd, sl], [out, tr])
    demo.load(replay, [dd, sl], [out, tr])

if __name__ == "__main__":
    demo.launch()
