"""Compare architectures using the saved outputs of each CV run.

Looks for these folders (any missing one is skipped):
    cv_out (resnet50), cv_out_densenet121, cv_out_efficientnet_b0
Run:  .venv/bin/python compare_archs.py [--root FOLDER_WITH_THE_cv_out_FOLDERS] [--quick]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from src.checks import check, summary  # noqa: E402

FOLDERS = {"resnet50": "cv_out", "densenet121": "cv_out_densenet121",
           "efficientnet_b0": "cv_out_efficientnet_b0"}


def ms(values):
    """mean and sample std (n-1) across folds."""
    v = np.array(values, float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else float("nan"))


def fmt(values):
    m, s = ms(values)
    return f"{m:.4f} +/- {s:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="folder that contains the cv_out* folders")
    ap.add_argument("--quick", action="store_true", help="use the *_quick folders")
    a = ap.parse_args()
    root = Path(a.root) if a.root else (Path("/kaggle/working") if Path("/kaggle/working").is_dir() else HERE)
    suffix = "_quick" if a.quick else ""

    # ---- load: probs[arch][fold][seed] = DataFrame(filename,label,prob)
    probs, seeds_of, n_folds = {}, {}, None
    for arch, folder in FOLDERS.items():
        d = root / (folder + suffix)
        if not (d / "results.json").exists():
            print(f"(skipping {arch}: {d / 'results.json'} not found)")
            continue
        res = json.load(open(d / "results.json"))
        seeds, nf = res["seeds"], res["n_folds"]
        n_folds = nf if n_folds is None else n_folds
        check(f"{arch}: same number of folds as the others", nf == n_folds, f"{nf}")
        probs[arch] = {k: {s: pd.read_csv(d / f"fold{k}_seed{s}" / "heldout_probs.csv") for s in seeds}
                       for k in range(nf)}
        seeds_of[arch] = seeds
        # cross-check my recomputation against what aggregate.py saved
        mine = [roc_auc_score(*_ens(probs[arch][k], seeds)) for k in range(nf)]
        saved = [r["ensemble"]["roc_auc"] for r in res["per_fold"]]
        check(f"{arch}: recomputed ensemble AUC matches results.json", np.allclose(mine, saved, atol=1e-9))
    if not probs:
        sys.exit("No results found.")
    archs = list(probs)
    if len(archs) > 1:  # the same images in the same order per fold -> fair paired comparison
        for k in range(n_folds):
            ref = probs[archs[0]][k][seeds_of[archs[0]][0]]["filename"]
            check(f"fold {k}: all architectures scored the same held-out images",
                  all(probs[x][k][seeds_of[x][0]]["filename"].equals(ref) for x in archs[1:]))

    # ---- per fold metrics
    rows = []
    for arch in archs:
        for k in range(n_folds):
            y, _ = _ens(probs[arch][k], seeds_of[arch])
            singles = [probs[arch][k][s]["prob"].values for s in seeds_of[arch]]
            ens = np.mean(singles, axis=0)
            rows.append({"arch": arch, "fold": k,
                         "single_roc": np.mean([roc_auc_score(y, p) for p in singles]),
                         "single_pr": np.mean([average_precision_score(y, p) for p in singles]),
                         "ens_roc": roc_auc_score(y, ens), "ens_pr": average_precision_score(y, ens)})
    pf = pd.DataFrame(rows)

    # ---- table 1: mean +/- std across folds
    t1 = pd.DataFrame([{"architecture": a_, **{c: fmt(pf[pf.arch == a_][c]) for c in
          ["single_roc", "ens_roc", "single_pr", "ens_pr"]}} for a_ in archs]).rename(columns={
          "single_roc": "single ROC-AUC", "ens_roc": "ensemble ROC-AUC",
          "single_pr": "single PR-AUC", "ens_pr": "ensemble PR-AUC"})
    print(f"\n=== Mean +/- std across {n_folds} folds (std uses n-1) ===")
    print(t1.to_string(index=False))

    # ---- table 2: paired per-fold differences vs resnet50
    t2 = None
    if "resnet50" in archs and len(archs) > 1:
        base = pf[pf.arch == "resnet50"].set_index("fold")
        out = []
        for a_ in archs[1:]:
            cur = pf[pf.arch == a_].set_index("fold")
            for c, name in [("single_roc", "single ROC-AUC"), ("ens_roc", "ensemble ROC-AUC"),
                            ("single_pr", "single PR-AUC"), ("ens_pr", "ensemble PR-AUC")]:
                d = (cur[c] - base[c]).values
                out.append({"architecture": a_, "metric": name, "mean diff vs resnet50": f"{d.mean():+.4f}",
                            "folds where diff > 0": f"{int((d > 0).sum())}/{len(d)}",
                            "per-fold diffs": " ".join(f"{x:+.3f}" for x in d)})
        t2 = pd.DataFrame(out)
        print("\n=== Paired per-fold differences vs resnet50 (same folds) ===")
        print(t2.to_string(index=False))

    # ---- table 3: EXPLORATORY all-architecture ensemble
    t3 = None
    if len(archs) > 1:
        e_roc, e_pr = [], []
        for k in range(n_folds):
            y, _ = _ens(probs[archs[0]][k], seeds_of[archs[0]])
            allp = [probs[x][k][s]["prob"].values for x in archs for s in seeds_of[x]]
            p = np.mean(allp, axis=0)
            e_roc.append(roc_auc_score(y, p)); e_pr.append(average_precision_score(y, p))
        print(f"\n=== EXPLORATORY: average of all {len(archs)} architectures' models ({', '.join(archs)}) ===")
        print("(not a pre-planned comparison; just a look at whether combining architectures helps)")
        print("per-fold ROC-AUC:", " ".join(f"{x:.4f}" for x in e_roc), "| mean +/- std:", fmt(e_roc))
        print("per-fold PR-AUC :", " ".join(f"{x:.4f}" for x in e_pr), "| mean +/- std:", fmt(e_pr))
        t3 = {"archs": archs, "roc_auc_per_fold": e_roc, "pr_auc_per_fold": e_pr,
              "roc_auc_mean_std": ms(e_roc), "pr_auc_mean_std": ms(e_pr)}

    # ---- save
    root.mkdir(exist_ok=True)
    t1.to_csv(root / f"compare_archs_table{suffix}.csv", index=False)
    if t2 is not None:
        t2.to_csv(root / f"compare_archs_paired{suffix}.csv", index=False)
    pf.to_csv(root / f"compare_archs_per_fold{suffix}.csv", index=False)
    json.dump({"per_fold": pf.to_dict("records"), "exploratory_all_arch_ensemble": t3},
              open(root / f"compare_archs{suffix}.json", "w"), indent=2, default=float)

    # ---- figure: per-fold ensemble ROC-AUC dots + mean bars
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), sharey=False)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for j, (col, title) in enumerate([("ens_roc", "Seed-ensemble ROC-AUC"), ("ens_pr", "Seed-ensemble PR-AUC")]):
        for i, a_ in enumerate(archs):
            v = pf[pf.arch == a_][col].values
            ax[j].bar(i, v.mean(), color=colors[i], alpha=0.35)
            ax[j].scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, color=colors[i], zorder=3)
            ax[j].errorbar(i, v.mean(), yerr=v.std(ddof=1) if len(v) > 1 else 0, color="k", capsize=4)
        ax[j].set(xticks=range(len(archs)), xticklabels=archs, title=title + " (dots = folds, bar = mean +/- std)")
        ax[j].tick_params(axis="x", labelrotation=15)
        ax[j].set_ylim(max(0, pf[col].min() - 0.08), min(1, pf[col].max() + 0.05))
    fig.tight_layout(); fig.savefig(root / f"compare_archs{suffix}.png", dpi=130); plt.close(fig)
    print(f"\nSaved tables, JSON and figure in {root}")
    sys.exit(0 if summary() else 1)


def _ens(fold_probs, seeds):
    """(labels, mean probability over seeds) for one fold of one architecture."""
    first = fold_probs[seeds[0]]
    return first["label"].values, np.mean([fold_probs[s]["prob"].values for s in seeds], axis=0)


if __name__ == "__main__":
    main()
