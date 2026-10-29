"""Model 2: GRU sequence classifier over the interleaved command-response events.

Consumes the ordered per-step encoding from features/sequence.py. Embeds the
categorical tokens (program, response category, timing bin), concatenates the two
scalar features, runs a GRU, and classifies from the final hidden state.

Evaluated with the SAME leakage-resistant grouped CV as Model 1, so the two are
directly comparable. Trains on CPU in a minute or so for the PoC.

Run:  python -m honeyai.models.recurrent
"""
from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from honeyai.schema import read_sessions
from honeyai.features.sequence import SequenceVocab
from honeyai.models.xgb_baseline import LABELS, LABEL_IDX, _groups

DATA = os.path.join("data", "sessions", "poc_dataset.jsonl")
MAX_LEN = 40
EPOCHS = 25
torch.manual_seed(0)


class GRUClassifier(nn.Module):
    def __init__(self, n_programs, n_resp=6, n_time=4, emb=16, hidden=48, n_classes=3):
        super().__init__()
        self.prog_emb = nn.Embedding(n_programs, emb, padding_idx=0)
        self.resp_emb = nn.Embedding(n_resp, 8)
        self.time_emb = nn.Embedding(n_time, 6)
        in_dim = emb + 8 + 6 + 2  # + cmd_len, progress scalars
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(hidden, n_classes))

    def forward(self, x, lengths):
        prog = self.prog_emb(x[:, :, 0].long())
        resp = self.resp_emb(x[:, :, 1].long())
        tim = self.time_emb(x[:, :, 2].long())
        scal = x[:, :, 3:5].float()
        feats = torch.cat([prog, resp, tim, scal], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            feats, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return self.head(h[-1])


def _encode_all(sessions, vocab):
    X, L = [], []
    for s in sessions:
        rows, length = vocab.encode(s, MAX_LEN)
        X.append(rows)
        L.append(max(length, 1))
    return np.array(X, dtype=np.float32), np.array(L)


def _train_fold(Xtr, Ltr, ytr, n_programs, class_w):
    model = GRUClassifier(n_programs)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(class_w, dtype=torch.float32))
    Xt = torch.tensor(Xtr); Lt = torch.tensor(Ltr); yt = torch.tensor(ytr)
    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        out = model(Xt, Lt)
        loss = lossf(out, yt)
        loss.backward()
        opt.step()
    return model


def run(n_splits=5, seed=0):
    sessions = read_sessions(DATA)
    vocab = SequenceVocab.build(sessions)
    X, L = _encode_all(sessions, vocab)
    y = np.array([LABEL_IDX[s["actor_label"]] for s in sessions])
    groups = np.array(_groups(sessions))

    # inverse-frequency class weights
    counts = np.bincount(y, minlength=3)
    class_w = (counts.sum() / (3 * np.maximum(counts, 1))).tolist()

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    f1s, all_true, all_pred = [], [], []
    for tr, te in sgkf.split(X, y, groups):
        model = _train_fold(X[tr], L[tr], y[tr], vocab.n_programs, class_w)
        model.eval()
        with torch.no_grad():
            out = model(torch.tensor(X[te]), torch.tensor(L[te]))
            pred = out.argmax(1).numpy()
        f1s.append(f1_score(y[te], pred, average="macro"))
        all_true.extend(y[te]); all_pred.extend(pred)

    print(f"Model 2 (GRU sequence) - leakage-resistant {n_splits}-fold")
    print(f"macro-F1: {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}\n")
    print(classification_report(all_true, all_pred, target_names=LABELS, digits=3))
    print("confusion matrix (rows=true, cols=pred):")
    print("        " + "  ".join(f"{l[:6]:>7}" for l in LABELS))
    cm = confusion_matrix(all_true, all_pred)
    for i, row in enumerate(cm):
        print(f"{LABELS[i][:6]:>6}  " + "  ".join(f"{v:>7}" for v in row))
    return float(np.mean(f1s))


if __name__ == "__main__":
    run()
