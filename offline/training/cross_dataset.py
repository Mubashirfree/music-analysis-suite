import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader, TensorDataset

from offline.config import DOCS_DIR
from offline.data.mood import MOOD_CLASSES
from offline.evaluation.mood_metrics import full_report, quadrant_from_va
from offline.training.mood_data import load_split
from offline.training.mood_net import MultitaskMLP
from offline.training.train_mood import evaluate, set_seed, val_score

ALPHAS = [1.0, 10.0, 100.0, 1000.0, 10000.0]
DIRECTIONS = [("deam", "pmemo"), ("pmemo", "deam")]
SEEDS = [42, 1, 2]
HIDDEN, DROPOUT, WD, LR = 256, 0.5, 0.1, 1e-3   # the settings chosen on validation
BATCH, EPOCHS, PATIENCE = 64, 300, 30
REG_W, CLS_W = 10.0, 1.0


def load_all():
    """All splits stacked: meta, X (clip vectors), y (valence/arousal), c (quadrant), split names."""
    metas, Xs, ys, cs, splits = [], [], [], [], []
    for s in ["train", "val", "test"]:
        m, X, y, c = load_split(s)
        metas.append(m)
        Xs.append(X)
        ys.append(y)
        cs.append(c.astype(np.int64))
        splits += [s] * len(m)
    return (pd.concat(metas, ignore_index=True), np.concatenate(Xs), np.concatenate(ys),
            np.concatenate(cs), np.array(splits))


def fit_mlp(tasks, seed, Z, y, c, tr, va, meta, device):
    """Train one MLP on the source training songs; pick the best epoch on source validation."""
    set_seed(seed)
    counts = np.maximum(np.bincount(c[tr], minlength=len(MOOD_CLASSES)), 1)
    class_w = torch.tensor(tr.sum() / (len(MOOD_CLASSES) * counts),
                           dtype=torch.float32, device=device)
    ds = TensorDataset(torch.from_numpy(Z[tr]), torch.from_numpy(y[tr]), torch.from_numpy(c[tr]))
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True)
    model = MultitaskMLP(hidden=HIDDEN, trunk_out=HIDDEN // 2, dropout=DROPOUT).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    meta_va = meta[va].reset_index(drop=True)

    best, best_state, bad = -1e9, None, 0
    for _ in range(EPOCHS):
        model.train()
        for xb, yb, cb in loader:
            xb, yb, cb = xb.to(device), yb.to(device), cb.to(device)
            opt.zero_grad(set_to_none=True)
            pred, logits = model(xb)
            loss = torch.zeros((), device=device)
            if tasks in ("both", "reg"):
                loss = loss + REG_W * F.smooth_l1_loss(pred, yb, beta=0.1)
            if tasks in ("both", "cls"):
                loss = loss + CLS_W * F.cross_entropy(logits, cb, weight=class_w)
            loss.backward()
            opt.step()
        report, _, _ = evaluate(model, Z[va], y[va], c[va], meta_va, tasks, device)
        score = val_score(report, tasks)
        if score > best:
            best, bad = score, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model


def summarise(reps) -> str:
    parts = []
    if "regression" in reps[0]:
        for dim, short in [("valence", "V"), ("arousal", "A")]:
            rmse = np.mean([r["regression"][dim]["rmse"] for r in reps])
            corr = np.mean([r["regression"][dim]["pearson_r"] for r in reps])
            parts.append(f"{short} rmse {rmse:.3f} r {corr:.3f}")
    if "classification" in reps[0]:
        f1 = [r["classification"]["macro_f1"] for r in reps]
        parts.append(f"macro-F1 {np.mean(f1):.3f} ± {np.std(f1):.3f}")
    return " | ".join(parts)


def print_model(label: str, entries: list) -> None:
    for name in ["in_domain", "cross"]:
        reps = [e[name] for e in entries]
        bias = [e[name + "_valence_bias"] for e in entries]
        extra = "" if bias[0] is None else f" | V bias {np.mean(bias):+.3f}"
        print(f"  {label:14s} {name:9s} {summarise(reps)}{extra}")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta, X, y, c, split = load_all()
    results = {}

    for source, target in DIRECTIONS:
        key = f"{source}_to_{target}"
        src = (meta["dataset"] == source).to_numpy()
        tgt = (meta["dataset"] == target).to_numpy()
        tr, va, te = src & (split == "train"), src & (split == "val"), src & (split == "test")
        print(f"\n== train on {source}, test on {target} ({tgt.sum()} songs); "
              f"in_domain = {source} test split ({te.sum()} songs)")

        mean = X[tr].mean(axis=0)
        std = X[tr].std(axis=0) + 1e-6
        Z = ((X - mean) / std).astype(np.float32)

        def entry_from(pred_fn):
            e = {}
            for name, mask in [("in_domain", te), ("cross", tgt)]:
                pred = pred_fn(mask)
                rep = full_report(meta[mask].reset_index(drop=True), y[mask], pred,
                                  c[mask], quadrant_from_va(pred))
                e[name] = rep["all"]
                e[name + "_valence_bias"] = round(float((pred[:, 0] - y[mask][:, 0]).mean()), 4)
            return e

        def mlp_entry(model, tasks):
            e = {}
            for name, mask in [("in_domain", te), ("cross", tgt)]:
                rep, va_pred, _ = evaluate(model, Z[mask], y[mask], c[mask],
                                           meta[mask].reset_index(drop=True), tasks, device)
                e[name] = rep["all"]
                e[name + "_valence_bias"] = (None if tasks == "cls" else
                                             round(float((va_pred[:, 0] - y[mask][:, 0]).mean()), 4))
            return e

        results[key] = {}

        mu = y[tr].mean(axis=0)
        entries = [entry_from(lambda m: np.tile(mu, (int(m.sum()), 1)))]
        results[key]["train_mean"] = entries
        print_model("train_mean", entries)

        best_alpha, best_score = None, 1e9
        for a in ALPHAS:
            r = Ridge(alpha=a).fit(Z[tr], y[tr])
            err = np.sqrt(((y[va] - np.clip(r.predict(Z[va]), 0, 1)) ** 2).mean(axis=0)).mean()
            if err < best_score:
                best_alpha, best_score = a, err
        ridge = Ridge(alpha=best_alpha).fit(Z[tr], y[tr])
        entries = [entry_from(lambda m: np.clip(ridge.predict(Z[m]), 0, 1))]
        results[key]["ridge"] = entries
        print_model("ridge", entries)

        for tasks, label in [("both", "mlp_multitask"), ("reg", "mlp_reg_only"), ("cls", "mlp_cls_only")]:
            entries = [mlp_entry(fit_mlp(tasks, seed, Z, y, c, tr, va, meta, device), tasks)
                       for seed in SEEDS]
            results[key][label] = entries
            print_model(label, entries)

    out = DOCS_DIR / "results" / "cross_dataset.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nsaved", out)


if __name__ == "__main__":
    main()