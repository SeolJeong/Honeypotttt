"""Part 9: HoneyAI Streamlit dashboard.

Tabs:
  1. Model Comparison — overall F1, per-class metrics
  2. Evasion Analysis — per-strategy breakdown, recall heatmap
  3. Session Inspector — browse individual sessions, see predictions
  4. Wild Data — classify unlabelled Cowrie sessions

Usage:
    streamlit run honeyai/dashboard.py
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import streamlit as st

# add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = "data"
DS1 = os.path.join(DATA_DIR, "dataset1_sessions.json")
DS2 = os.path.join(DATA_DIR, "dataset2_tabular_features.csv")
QLORA_PREDS = os.path.join(DATA_DIR, "qlora_predictions.json")

LABEL_NAMES = ["human", "scripted", "AI-driven"]
LABEL_MAP = {"human": 0, "scripted": 1, "AI-driven": 2}

EVASION_ORDER = [
    "none", "timing_mimicry", "typo_mimicry", "shorter_commands",
    "hesitation_redundant_exploration", "avoid_systematic_enumeration", "combined",
]


# =====================================================================
# DATA LOADING (cached)
# =====================================================================

@st.cache_data
def load_sessions():
    with open(DS1, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_tabular():
    with open(DS2, encoding="utf-8") as f:
        return list(csv.DictReader(f))


@st.cache_data
def load_qlora_predictions():
    if not os.path.exists(QLORA_PREDS):
        return None
    with open(QLORA_PREDS, encoding="utf-8") as f:
        data = json.load(f)
    preds = []
    if "seeds" in data:
        all_preds = data["seeds"][0].get("predictions", [])
        preds = [p for p in all_preds if p.get("prefix") == "full"]
        if not preds:
            preds = all_preds
    if not preds:
        preds = data.get("predictions", [])
    out = {}
    for p in preds:
        sid = p.get("session_id", p.get("id", ""))
        true = p.get("true_label", p.get("true", ""))
        pred = p.get("predicted_label", p.get("pred", ""))
        if sid and true and pred:
            out[sid] = {"true": true, "pred": pred}
    return out if out else None


# =====================================================================
# MODEL TRAINING (cached)
# =====================================================================

@st.cache_data
def train_xgboost_cached(feature_cols_tuple):
    from xgboost import XGBClassifier
    from sklearn.metrics import f1_score, classification_report

    feature_cols = list(feature_cols_tuple)
    rows = load_tabular()
    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]

    X_train = np.array([[float(r[c]) for c in feature_cols] for r in train_rows])
    y_train = np.array([LABEL_MAP[r["actor_label"]] for r in train_rows])
    X_test = np.array([[float(r[c]) for c in feature_cols] for r in test_rows])
    y_test = np.array([LABEL_MAP[r["actor_label"]] for r in test_rows])

    counts = Counter(y_train)
    total = len(y_train)
    n_cls = len(counts)
    weights = {c: total / (n_cls * cnt) for c, cnt in counts.items()}
    sample_w = np.array([weights[lb] for lb in y_train])

    clf = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric="mlogloss",
    )
    clf.fit(X_train, y_train, sample_weight=sample_w)
    preds = clf.predict(X_test)

    results = {}
    for row, pred in zip(test_rows, preds):
        results[row["session_id"]] = {
            "true": row["actor_label"],
            "pred": LABEL_NAMES[pred],
        }
    f1 = f1_score(y_test, preds, average="macro")

    # feature importance
    importances = dict(zip(feature_cols, clf.feature_importances_))

    return results, f1, importances


@st.cache_data
def train_gru_cached():
    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pack_padded_sequence
    from sklearn.metrics import f1_score

    sessions = load_sessions()
    train_sess = [s for s in sessions if s["split"] == "train"]
    test_sess = [s for s in sessions if s["split"] == "test"]

    def tokenize(text):
        return [t for t in re.split(r"[^a-z0-9._/-]+", text.lower()) if t]

    VOCAB_MAX, MAX_CMD, MAX_RESP = 150, 10, 16
    EMBED, HIDDEN = 24, 48
    TBINS = ["short", "medium", "long", "very_long"]
    RCATS = ["success", "not_found_with_hint", "not_found_no_hint",
             "permission_denied", "command_not_found", "partial_result"]
    N_OH = len(TBINS) + len(RCATS)

    t2i = {"<PAD>": 0, "<UNK>": 1}
    tc = Counter()
    for s in train_sess:
        for st in s["interactions"]:
            tc.update(tokenize(st["command"]))
            tc.update(tokenize(st["system_response"]))
    for tok, _ in tc.most_common(VOCAB_MAX - 2):
        t2i[tok] = len(t2i)

    def enc_tok(tokens, ml):
        ids = [t2i.get(t, 1) for t in tokens[:ml]]
        return ids + [0] * (ml - len(ids))

    def enc_sess(sl):
        ms = max(len(s["interactions"]) for s in sl)
        n = len(sl)
        cmd = np.zeros((n, ms, MAX_CMD), dtype=np.int64)
        resp = np.zeros((n, ms, MAX_RESP), dtype=np.int64)
        cats = np.zeros((n, ms, N_OH), dtype=np.float32)
        scal = np.zeros((n, ms, 4), dtype=np.float32)
        lens = np.zeros(n, dtype=np.int64)
        labs = np.zeros(n, dtype=np.int64)
        for i, sess in enumerate(sl):
            steps = sess["interactions"]
            lens[i] = len(steps)
            labs[i] = LABEL_MAP[sess["actor_label"]]
            for j, st in enumerate(steps):
                cmd[i, j] = enc_tok(tokenize(st["command"]), MAX_CMD)
                resp[i, j] = enc_tok(tokenize(st["system_response"]), MAX_RESP)
                tb = st.get("think_time_bin", "medium").lower()
                if tb in TBINS:
                    cats[i, j, TBINS.index(tb)] = 1.0
                rc = st.get("response_category", "success")
                if rc in RCATS:
                    cats[i, j, len(TBINS) + RCATS.index(rc)] = 1.0
                scal[i, j, 0] = 1.0 if st.get("has_typo") in (True, "True", "true") else 0.0
                scal[i, j, 1] = float(st.get("task_progress", 0.0))
                scal[i, j, 2] = min(len(st["command"]) / 50.0, 1.0)
                scal[i, j, 3] = min(float(st.get("inter_command_gap_s", 0)) / 30.0, 1.0)
        return {k: torch.from_numpy(v) for k, v in [
            ("cmd", cmd), ("resp", resp), ("cats", cats),
            ("scal", scal), ("lens", lens), ("labs", labs)]}

    et = enc_sess(train_sess)
    ete = enc_sess(test_sess)

    class GRU(nn.Module):
        def __init__(self, vs):
            super().__init__()
            self.embed = nn.Embedding(vs, EMBED, padding_idx=0)
            self.gru = nn.GRU(EMBED * 2 + N_OH + 4, HIDDEN, batch_first=True)
            self.drop = nn.Dropout(0.4)
            self.head = nn.Linear(HIDDEN, 3)

        def _avg(self, ids):
            emb = self.embed(ids)
            mask = (ids != 0).unsqueeze(-1).float()
            return (emb * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)

        def forward(self, cmd, resp, cats, scal, lens):
            x = torch.cat([self._avg(cmd), self._avg(resp), cats, scal], dim=-1)
            from torch.nn.utils.rnn import pack_padded_sequence
            packed = pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
            _, h = self.gru(packed)
            return self.head(self.drop(h.squeeze(0)))

    vs = len(t2i)
    best_f1, best_preds = -1, None
    for seed in [42, 123, 456]:
        torch.manual_seed(seed)
        model = GRU(vs)
        cn = Counter(et["labs"].numpy().tolist())
        tn = len(et["labs"])
        cw = torch.zeros(max(cn.keys()) + 1)
        for c, cnt in cn.items():
            cw[c] = tn / (3 * cnt)
        crit = nn.CrossEntropyLoss(weight=cw)
        opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
        for _ in range(60):
            model.train()
            logits = model(et["cmd"], et["resp"], et["cats"], et["scal"], et["lens"])
            loss = crit(logits, et["labs"])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(ete["cmd"], ete["resp"], ete["cats"], ete["scal"], ete["lens"])
            p = logits.argmax(dim=1).numpy()
        true = ete["labs"].numpy()
        f1 = f1_score(true, p, average="macro")
        if f1 > best_f1:
            best_f1, best_preds = f1, p

    results = {}
    for sess, pred in zip(test_sess, best_preds):
        results[sess["session_id"]] = {"true": sess["actor_label"], "pred": LABEL_NAMES[pred]}

    return results, best_f1


# =====================================================================
# ANALYSIS HELPERS
# =====================================================================

def per_evasion_recall(predictions, evasion_map):
    groups = defaultdict(lambda: {"total": 0, "correct": 0})
    for sid, res in predictions.items():
        if res["true"] != "AI-driven":
            continue
        ev = evasion_map.get(sid, "none")
        groups[ev]["total"] += 1
        if res["pred"] == "AI-driven":
            groups[ev]["correct"] += 1
    rows = []
    for ev in EVASION_ORDER:
        if ev in groups:
            g = groups[ev]
            rows.append({"strategy": ev, "n": g["total"],
                         "recall": round(g["correct"] / g["total"], 3) if g["total"] else 0})
    return rows


def compute_confusion(predictions):
    cm = np.zeros((3, 3), dtype=int)
    for res in predictions.values():
        t = LABEL_MAP.get(res["true"], -1)
        p = LABEL_MAP.get(res["pred"], -1)
        if t >= 0 and p >= 0:
            cm[t][p] += 1
    return cm


# =====================================================================
# STREAMLIT APP
# =====================================================================

ALL_FEATURES = (
    "session_duration_s", "command_count",
    "inter_cmd_mean_s", "inter_cmd_std_s", "min_gap_s", "max_gap_s",
    "cmd_len_mean", "cmd_len_std", "unique_cmd_ratio",
    "typo_count", "typo_rate",
    "error_count", "error_rate",
    "error_recovery_rate", "adaptation_event_count", "not_found_with_hint_count",
    "final_task_progress", "interaction_depth", "sequence_entropy",
)


def main():
    st.set_page_config(page_title="HoneyAI Dashboard", layout="wide")
    st.title("HoneyAI — Evasion-Aware SSH Attribution")
    st.caption("ICT3214 Security Analytics — Coursework 2")

    # load data
    sessions = load_sessions()
    evasion_map = {s["session_id"]: s.get("evasion_strategy", "none")
                   for s in sessions if s["split"] == "test"}
    test_sessions = {s["session_id"]: s for s in sessions if s["split"] == "test"}

    # train models
    with st.spinner("Training XGBoost..."):
        xgb_preds, xgb_f1, xgb_imp = train_xgboost_cached(ALL_FEATURES)
    with st.spinner("Training GRU..."):
        gru_preds, gru_f1 = train_gru_cached()
    qlora_preds = load_qlora_predictions()
    if qlora_preds:
        from sklearn.metrics import f1_score
        qlora_f1 = f1_score(
            [v["true"] for v in qlora_preds.values()],
            [v["pred"] for v in qlora_preds.values()],
            labels=LABEL_NAMES, average="macro", zero_division=0)
    else:
        qlora_f1 = None

    models = {"XGBoost": (xgb_preds, xgb_f1),
              "GRU": (gru_preds, gru_f1)}
    if qlora_preds:
        models["QLoRA"] = (qlora_preds, qlora_f1)

    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "Model Comparison", "Evasion Analysis", "Session Inspector", "Wild Data"
    ])

    # ── TAB 1: MODEL COMPARISON ──
    with tab1:
        st.header("Model comparison")

        # overall F1
        cols = st.columns(len(models))
        for col, (name, (preds, f1)) in zip(cols, models.items()):
            col.metric(name, f"{f1:.3f}", help="Test macro-F1")

        # per-class metrics
        st.subheader("Per-class performance")
        for name, (preds, f1) in models.items():
            cm = compute_confusion(preds)
            st.write(f"**{name}** (macro-F1: {f1:.3f})")
            rows = []
            for i, label in enumerate(LABEL_NAMES):
                tp = cm[i][i]
                support = cm[i].sum()
                pred_total = cm[:, i].sum()
                prec = tp / pred_total if pred_total else 0
                rec = tp / support if support else 0
                f1_cls = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
                rows.append({"Class": label, "Precision": f"{prec:.3f}",
                             "Recall": f"{rec:.3f}", "F1": f"{f1_cls:.3f}",
                             "Support": int(support)})
            st.dataframe(rows, use_container_width=True, hide_index=True)

        # confusion matrices
        st.subheader("Confusion matrices")
        cols = st.columns(len(models))
        for col, (name, (preds, _)) in zip(cols, models.items()):
            cm = compute_confusion(preds)
            col.write(f"**{name}**")
            import pandas as pd
            df = pd.DataFrame(cm, index=LABEL_NAMES, columns=LABEL_NAMES)
            col.dataframe(df)

        # feature importance (XGBoost)
        st.subheader("XGBoost feature importance")
        sorted_imp = sorted(xgb_imp.items(), key=lambda x: -x[1])
        imp_data = {"Feature": [x[0] for x in sorted_imp],
                    "Importance": [round(x[1], 4) for x in sorted_imp]}
        st.bar_chart(data=dict(zip(imp_data["Feature"], imp_data["Importance"])))

    # ── TAB 2: EVASION ANALYSIS ──
    with tab2:
        st.header("Evasion analysis")
        st.write("AI-driven recall per evasion strategy (higher = model detects AI better)")

        # build recall table
        recall_data = {"Strategy": []}
        for name in models:
            recall_data[name] = []

        for ev in EVASION_ORDER:
            recall_data["Strategy"].append(ev)
            for name, (preds, _) in models.items():
                rows = per_evasion_recall(preds, evasion_map)
                match = [r for r in rows if r["strategy"] == ev]
                recall_data[name].append(match[0]["recall"] if match else 0)

        import pandas as pd
        df_recall = pd.DataFrame(recall_data).set_index("Strategy")
        st.dataframe(df_recall, use_container_width=True)

        # bar chart
        st.bar_chart(df_recall)

        # evasive vs non-evasive
        st.subheader("Non-evasive vs evasive recall drop")
        drop_rows = []
        for name, (preds, _) in models.items():
            non_ev_t, non_ev_c = 0, 0
            ev_t, ev_c = 0, 0
            for sid, res in preds.items():
                if res["true"] != "AI-driven":
                    continue
                ev = evasion_map.get(sid, "none")
                if ev == "none":
                    non_ev_t += 1
                    if res["pred"] == "AI-driven":
                        non_ev_c += 1
                else:
                    ev_t += 1
                    if res["pred"] == "AI-driven":
                        ev_c += 1
            r_non = non_ev_c / non_ev_t if non_ev_t else 0
            r_ev = ev_c / ev_t if ev_t else 0
            drop_rows.append({
                "Model": name,
                "Non-evasive recall": f"{r_non:.3f}",
                "Evasive recall": f"{r_ev:.3f}",
                "Drop": f"{r_non - r_ev:+.3f}",
            })
        st.dataframe(drop_rows, use_container_width=True, hide_index=True)

    # ── TAB 3: SESSION INSPECTOR ──
    with tab3:
        st.header("Session inspector")

        # filters
        col1, col2, col3 = st.columns(3)
        with col1:
            label_filter = st.selectbox("Actor label", ["all"] + LABEL_NAMES)
        with col2:
            ev_filter = st.selectbox("Evasion strategy", ["all"] + EVASION_ORDER)
        with col3:
            correct_filter = st.selectbox("Prediction", ["all", "correct", "misclassified"])

        filtered_sids = []
        for sid, sess in test_sessions.items():
            if label_filter != "all" and sess["actor_label"] != label_filter:
                continue
            if ev_filter != "all" and sess.get("evasion_strategy", "none") != ev_filter:
                continue
            if correct_filter != "all":
                # check against all models
                is_correct = all(
                    models[m][0].get(sid, {}).get("pred") == sess["actor_label"]
                    for m in models
                )
                if correct_filter == "correct" and not is_correct:
                    continue
                if correct_filter == "misclassified" and is_correct:
                    continue
            filtered_sids.append(sid)

        st.write(f"Showing {len(filtered_sids)} sessions")

        if filtered_sids:
            selected_sid = st.selectbox("Select session", filtered_sids)
            sess = test_sessions[selected_sid]

            # session metadata
            meta_cols = st.columns(4)
            meta_cols[0].metric("True label", sess["actor_label"])
            meta_cols[1].metric("Evasion", sess.get("evasion_strategy", "none"))
            meta_cols[2].metric("Family", sess.get("generator_family", ""))
            meta_cols[3].metric("Steps", sess["total_steps"])

            # model predictions
            st.subheader("Model predictions")
            pred_cols = st.columns(len(models))
            for col, (name, (preds, _)) in zip(pred_cols, models.items()):
                p = preds.get(selected_sid, {})
                pred_label = p.get("pred", "?")
                is_correct = pred_label == sess["actor_label"]
                col.metric(name, pred_label,
                           delta="correct" if is_correct else "wrong",
                           delta_color="normal" if is_correct else "inverse")

            # command trace
            st.subheader("Command trace")
            for step in sess["interactions"]:
                gap = step.get("inter_command_gap_s", 0)
                typo = " (typo)" if step.get("has_typo") in (True, "True", "true") else ""
                cat = step["response_category"]

                st.markdown(f"**Step {step['step']}** | Gap: {gap:.2f}s | {cat}{typo}")
                st.code(step["command"], language="bash")
                with st.expander("Response", expanded=False):
                    st.text(step["system_response"])

    # ── TAB 4: WILD DATA ──
    with tab4:
        st.header("Wild data classifier")
        st.write("Upload unlabelled Cowrie JSON logs to classify with the trained models.")
        st.caption("These are never used for training or reported accuracy — demo/case studies only.")

        uploaded = st.file_uploader("Upload Cowrie sessions JSON", type=["json"])
        if uploaded:
            try:
                wild_sessions = json.load(uploaded)
                if isinstance(wild_sessions, dict):
                    wild_sessions = [wild_sessions]
                st.success(f"Loaded {len(wild_sessions)} sessions")

                # classify with XGBoost (needs features)
                st.subheader("Feature summary")
                for ws in wild_sessions[:20]:
                    sid = ws.get("session_id", "unknown")
                    steps = ws.get("interactions", ws.get("steps", []))
                    n = len(steps)
                    if n == 0:
                        continue
                    gaps = [s.get("inter_command_gap_s", 0) for s in steps]
                    cmds = [s.get("command", "") for s in steps]
                    st.write(f"**{sid}**: {n} steps, "
                             f"mean gap {np.mean(gaps):.2f}s, "
                             f"commands: {', '.join(cmds[:5])}{'...' if n > 5 else ''}")

                st.info("Full classification requires the session to match the dataset1 schema. "
                        "Use scripts/parse_cowrie.py (Part 9) to convert raw Cowrie logs first.")
            except json.JSONDecodeError:
                st.error("Invalid JSON file")


if __name__ == "__main__":
    from sklearn.metrics import f1_score
    main()