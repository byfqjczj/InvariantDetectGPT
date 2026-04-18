"""
Proper threshold evaluation: split held-out samples into val + test,
calibrate threshold on val, report accuracy/F1/confusion on test.

Uses pre-computed scores from comparison_results.json.

Usage:
  python evaluation/threshold_eval.py
"""

import json
import os
import random
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, roc_auc_score
)

RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comparison_results.json")
SEED = 42
VAL_RATIO = 0.5

TRAIN_GENERATORS = {"mistral-7b", "qwen-7b"}
HELD_GENERATORS  = {"gemma-7b", "phi-3-mini", "deepseek-7b"}


def find_optimal_threshold(scores, labels, num_thresholds=1000):
    best_t, best_f1 = 0.0, 0.0
    for t in np.linspace(0, 1, num_thresholds):
        preds = (np.array(scores) >= t).astype(int)
        tp = np.sum((preds == 1) & (np.array(labels) == 1))
        fp = np.sum((preds == 1) & (np.array(labels) == 0))
        fn = np.sum((preds == 0) & (np.array(labels) == 1))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def print_eval(name, labels, scores, threshold):
    preds = (np.array(scores) >= threshold).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc   = accuracy_score(labels, preds)
    f1    = f1_score(labels, preds, zero_division=0)
    auroc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else float("nan")
    fpr   = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr   = fn / (fn + tp) if (fn + tp) > 0 else 0

    print(f"\n  {name}  (threshold={threshold:.4f})")
    print(f"  {'':20} Pred Human   Pred AI")
    print(f"  {'True Human':20} {tn:>10}   {fp:>7}")
    print(f"  {'True AI':20} {fn:>10}   {tp:>7}")
    print(f"  Acc={acc:.3f}  F1={f1:.3f}  AUROC={auroc:.4f}  FPR={fpr:.3f}  FNR={fnr:.3f}")

    return {"threshold": threshold, "accuracy": acc, "f1": f1, "auroc": auroc,
            "fpr": fpr, "fnr": fnr, "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


def evaluate_split(title, samples):
    if not samples:
        return None

    random.seed(SEED)
    shuffled = samples[:]
    random.shuffle(shuffled)
    split = int(len(shuffled) * VAL_RATIO)
    val, test = shuffled[:split], shuffled[split:]

    val_labels  = [s["label"] for s in val]
    test_labels = [s["label"] for s in test]

    if len(set(val_labels)) < 2 or len(set(test_labels)) < 2:
        print(f"\n  [{title}] skipped — not enough class diversity after split")
        return None

    val_el  = [s["editlens_score"] for s in val]
    val_our = [s["our_score"] for s in val]
    test_el  = [s["editlens_score"] for s in test]
    test_our = [s["our_score"] for s in test]

    el_thresh,  el_val_f1  = find_optimal_threshold(val_el,  val_labels)
    our_thresh, our_val_f1 = find_optimal_threshold(val_our, val_labels)

    print(f"\n{'=' * 60}")
    print(f"{title}  (val={len(val)}, test={len(test)})")
    print(f"  Val F1 — EditLens: {el_val_f1:.3f} @ {el_thresh:.4f} | "
          f"Ours: {our_val_f1:.3f} @ {our_thresh:.4f}")
    print(f"{'=' * 60}")

    el_result  = print_eval("EditLens RoBERTa", test_labels, test_el,  el_thresh)
    our_result = print_eval("Ours            ", test_labels, test_our, our_thresh)

    return {
        "n_val": len(val),
        "n_test": len(test),
        "editlens": {**el_result,  "val_f1": el_val_f1},
        "ours":     {**our_result, "val_f1": our_val_f1},
    }


def main():
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    samples = results["samples"]
    all_results = {}

    all_results["overall"] = evaluate_split("OVERALL", samples)

    for domain in sorted({s["domain"] for s in samples}):
        all_results[f"domain_{domain}"] = evaluate_split(
            f"Domain: {domain.upper()}",
            [s for s in samples if s["domain"] == domain]
        )

    all_results["held_out_generators"] = evaluate_split(
        "HELD-OUT GENERATORS (gemma, phi-3-mini, deepseek)",
        [s for s in samples if s["source_model"] in HELD_GENERATORS or s["source_model"] == "human"]
    )

    for src in sorted({s["source_model"] for s in samples if s["source_model"] != "human"}):
        all_results[f"src_{src}"] = evaluate_split(
            f"human vs {src}",
            [s for s in samples if s["source_model"] in (src, "human")]
        )

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "threshold_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
