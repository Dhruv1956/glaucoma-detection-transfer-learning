"""Turn the saved probabilities into numbers, confidence intervals and figures."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, precision_recall_curve, roc_auc_score,
                             roc_curve)

from .checks import check


def _auc(y, p):
    """ROC-AUC, or NaN if only one class is present (can happen in tiny QUICK_TEST folds)."""
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


def _ap(y, p):
    return average_precision_score(y, p) if len(np.unique(y)) == 2 else np.nan


def youden_threshold(y, p):
    """Threshold maximising sensitivity + specificity - 1, chosen on the INNER-val set."""
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, p)
    t = thr[np.argmax(tpr - fpr)]
    return 0.5 if not np.isfinite(t) else float(t)


def sens_spec_prec(y, p, thr):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    return {"sensitivity": tp / (tp + fn) if tp + fn else np.nan,
            "specificity": tn / (tn + fp) if tn + fp else np.nan,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "tp": tp, "fn": fn, "tn": tn, "fp": fp}


def _mean_std(values):
    v = np.array(values, dtype=float)
    return {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v, ddof=1)) if np.sum(~np.isnan(v)) > 1 else float("nan")}


def _fmt(d):
    return f"{d['mean']:.4f} +/- {d['std']:.4f}"


def aggregate(cfg):
    out = cfg.out_dir
    print("\n=== Aggregation ===")
    info = pd.read_csv(out / "folds.csv", dtype={"group_id": str})
    group_of = dict(zip(info["filename"], info["group_id"]))
    seeds = list(cfg.seeds)

    # ---- load every (fold, seed) result
    held, inner = {}, {}
    for k in range(cfg.n_folds):
        for s in seeds:
            d = out / f"fold{k}_seed{s}"
            held[k, s] = pd.read_csv(d / "heldout_probs.csv")
            inner[k, s] = pd.read_csv(d / "innerval_probs.csv")
    for k in range(cfg.n_folds):  # all seeds of a fold must score the same images in the same order
        for s in seeds[1:]:
            check(f"fold {k}: seeds score identical held-out/inner-val images",
                  held[k, s]["filename"].equals(held[k, seeds[0]]["filename"])
                  and inner[k, s]["filename"].equals(inner[k, seeds[0]]["filename"]))

    # ---- per fold / per seed / ensemble metrics (ensemble = mean of the seeds' probabilities)
    per_fold = []
    for k in range(cfg.n_folds):
        y = held[k, seeds[0]]["label"].values
        yi = inner[k, seeds[0]]["label"].values
        probs = {s: held[k, s]["prob"].values for s in seeds}
        ens = np.mean([probs[s] for s in seeds], axis=0)
        ens_inner = np.mean([inner[k, s]["prob"].values for s in seeds], axis=0)
        row = {"fold": k, "n": int(len(y)), "n_glaucoma": int(y.sum()), "seeds": {}}
        for s in seeds:
            thr = youden_threshold(yi, inner[k, s]["prob"].values)   # threshold from inner-val only
            row["seeds"][s] = {"roc_auc": _auc(y, probs[s]), "pr_auc": _ap(y, probs[s]), "threshold": thr,
                               **sens_spec_prec(y, probs[s], thr)}
        thr_e = youden_threshold(yi, ens_inner)
        row["ensemble"] = {"roc_auc": _auc(y, ens), "pr_auc": _ap(y, ens), "threshold": thr_e,
                           **sens_spec_prec(y, ens, thr_e)}
        per_fold.append(row)
        print(f"fold {k}: " + " | ".join(f"seed{s} AUC={row['seeds'][s]['roc_auc']:.4f}" for s in seeds)
              + f" | ENSEMBLE AUC={row['ensemble']['roc_auc']:.4f} PR-AUC={row['ensemble']['pr_auc']:.4f}")

    # ---- mean +/- std across folds: single (averaged over seeds within a fold) vs ensemble
    metrics = ["roc_auc", "pr_auc", "sensitivity", "specificity", "precision"]
    across = {"single": {}, "ensemble": {}}
    for m in metrics:
        across["single"][m] = _mean_std([np.nanmean([r["seeds"][s][m] for s in seeds]) for r in per_fold])
        across["ensemble"][m] = _mean_std([r["ensemble"][m] for r in per_fold])
    print("\nMean +/- std across folds (std uses n-1):")
    for m in metrics:
        print(f"  {m:12s} single={_fmt(across['single'][m])}   ensemble={_fmt(across['ensemble'][m])}")

    # ---- pooled out-of-fold predictions + bootstrap (resampling whole groups)
    pooled = pd.concat([held[k, seeds[0]][["filename", "label"]] for k in range(cfg.n_folds)], ignore_index=True)
    for s in seeds:
        pooled[f"p{s}"] = np.concatenate([held[k, s]["prob"].values for k in range(cfg.n_folds)])
    pooled["ens"] = pooled[[f"p{s}" for s in seeds]].mean(axis=1)
    pooled["group"] = pooled["filename"].map(group_of)
    y_all = pooled["label"].values

    def scores(idx):  # single = mean over seeds of the per-seed metric; ensemble = metric of mean prob
        y = y_all[idx]
        r = {"single_roc": np.mean([_auc(y, pooled[f"p{s}"].values[idx]) for s in seeds]),
             "single_pr": np.mean([_ap(y, pooled[f"p{s}"].values[idx]) for s in seeds]),
             "ens_roc": _auc(y, pooled["ens"].values[idx]), "ens_pr": _ap(y, pooled["ens"].values[idx])}
        r["diff_roc"] = r["ens_roc"] - r["single_roc"]
        r["diff_pr"] = r["ens_pr"] - r["single_pr"]
        return r

    point = scores(np.arange(len(pooled)))
    codes = pd.factorize(pooled["group"])[0]
    members = [np.where(codes == g)[0] for g in range(codes.max() + 1)]
    rng = np.random.default_rng(42)
    boots, skipped = [], 0
    for _ in range(cfg.n_bootstrap):
        pick = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[g] for g in pick])
        if len(np.unique(y_all[idx])) < 2:
            skipped += 1
            continue
        boots.append(scores(idx))
    boots = pd.DataFrame(boots)
    pooled_res = {}
    for key in point:
        lo, hi = np.nanpercentile(boots[key], [2.5, 97.5])
        pooled_res[key] = {"estimate": float(point[key]), "ci95": [float(lo), float(hi)]}
    pooled_res["bootstrap_resamples_used"] = int(len(boots))
    pooled_res["bootstrap_resamples_skipped_single_class"] = skipped
    print(f"\nPooled out-of-fold (n={len(pooled)}, {len(members)} groups), 95% bootstrap CI "
          f"({len(boots)} resamples by group_id):")
    for key, label in [("single_roc", "single ROC-AUC"), ("ens_roc", "ensemble ROC-AUC"),
                       ("single_pr", "single PR-AUC"), ("ens_pr", "ensemble PR-AUC"),
                       ("diff_roc", "ens - single ROC"), ("diff_pr", "ens - single PR")]:
        r = pooled_res[key]
        print(f"  {label:18s} {r['estimate']:.4f}  [{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}]")

    # ---- pooled thresholded metrics (each fold used its own inner-val threshold)
    def pool_conf(getter):
        c = {k: sum(getter(r)[k] for r in per_fold) for k in ("tp", "fn", "tn", "fp")}
        return {"sensitivity": c["tp"] / max(c["tp"] + c["fn"], 1), "specificity": c["tn"] / max(c["tn"] + c["fp"], 1),
                "precision": c["tp"] / max(c["tp"] + c["fp"], 1), **c}
    thr_pooled = {"ensemble": pool_conf(lambda r: r["ensemble"]),
                  "single_per_seed": {str(s): pool_conf(lambda r, s=s: r["seeds"][s]) for s in seeds}}
    print("\nAt the inner-val Youden threshold (pooled over folds), ensemble: "
          + ", ".join(f"{m}={thr_pooled['ensemble'][m]:.3f}" for m in ("sensitivity", "specificity", "precision")))

    # ---- run logs
    logs = [json.load(open(out / f"fold{k}_seed{s}" / "log.json")) for k in range(cfg.n_folds) for s in seeds]
    time_summary = {"mean_seconds_per_run": float(np.mean([l["seconds"] for l in logs])),
                    "total_seconds": float(np.sum([l["seconds"] for l in logs])),
                    "phase_chosen_counts": pd.Series([l["phase_chosen"] for l in logs]).value_counts().to_dict()}
    print("Runs:", time_summary)

    # ---- figures
    colors = {"ens": "#1f77b4", "single": "#c0c0c0"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for s in seeds:
        if len(np.unique(y_all)) == 2:
            f, t, _ = roc_curve(y_all, pooled[f"p{s}"]); ax[0].plot(f, t, color=colors["single"], lw=1)
            p, r, _ = precision_recall_curve(y_all, pooled[f"p{s}"]); ax[1].plot(r, p, color=colors["single"], lw=1)
    if len(np.unique(y_all)) == 2:
        f, t, _ = roc_curve(y_all, pooled["ens"]); ax[0].plot(f, t, color=colors["ens"], lw=2.2,
            label=f"ensemble AUC={point['ens_roc']:.3f}")
        p, r, _ = precision_recall_curve(y_all, pooled["ens"]); ax[1].plot(r, p, color=colors["ens"], lw=2.2,
            label=f"ensemble AP={point['ens_pr']:.3f}")
    ax[0].plot([0, 1], [0, 1], "k:", lw=1); ax[0].set(xlabel="False positive rate", ylabel="True positive rate",
        title="Pooled out-of-fold ROC (grey = single seeds)"); ax[0].legend(loc="lower right")
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Pooled out-of-fold PR"); ax[1].legend(loc="lower left")
    fig.tight_layout(); fig.savefig(out / "fig_pooled_roc_pr.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = np.arange(cfg.n_folds)
    ax.plot(x, [np.nanmean([r["seeds"][s]["roc_auc"] for s in seeds]) for r in per_fold], "o-", color="#808080", label="single (mean of seeds)")
    ax.plot(x, [r["ensemble"]["roc_auc"] for r in per_fold], "s-", color=colors["ens"], label="seed ensemble")
    ax.set(xticks=x, xlabel="Fold", ylabel="Held-out ROC-AUC", title="Per-fold ROC-AUC"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "fig_per_fold_auc.png", dpi=130); plt.close(fig)

    results = {"init": cfg.init_tag or "imagenet", "pretrain_seed": cfg.pretrain_seed if cfg.init_weights else None, "arch": cfg.arch, "img_size": cfg.img_size, "quick_test": cfg.quick_test, "n_images": int(len(pooled)), "n_folds": cfg.n_folds, "seeds": seeds,
               "per_fold": [{**r, "seeds": {str(s): v for s, v in r["seeds"].items()}} for r in per_fold],
               "mean_std_across_folds": across, "pooled_out_of_fold": pooled_res,
               "pooled_at_inner_val_threshold": thr_pooled, "runs": time_summary}
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Saved results.json and figures in {out}")
    return results
