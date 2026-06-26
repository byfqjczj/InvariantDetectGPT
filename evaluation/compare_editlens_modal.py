"""
Compare EditLens RoBERTa vs our domain-split detector on the Wikipedia/SQuAD domain.

Train domain split: trained on news+creative, never saw wikipedia.
EditLens: also never trained on our wikipedia samples.
=> Fair apples-to-apples comparison on a held-out domain.

Usage:
  cd implementation
  PYTHONUTF8=1 modal run evaluation/compare_editlens_modal.py

Output:
  evaluation/squad_comparison_results.json
"""

import json
import os
import modal

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGG_PATH     = os.path.join(BASE_DIR, "features", "extracted", "aggregated.json")
RAW_DIR      = os.path.join(BASE_DIR, "data", "raw")
GEN_DIR      = os.path.join(BASE_DIR, "generation", "generated")
DETECTOR_PT  = os.path.join(BASE_DIR, "training", "detector_domain_split.pt")
OUT_PATH     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "squad_comparison_results.json")

EDITLENS_CHECKPOINT = "pangram/editlens_roberta-large"
EDITLENS_BASE_MODEL = "FacebookAI/roberta-large"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.0",
        "transformers==4.44.0",
        "datasets==2.20.0",
        "peft==0.11.0",
        "safetensors",
        "scipy",
        "scikit-learn",
        "numpy",
        "huggingface_hub",
        "accelerate",
    )
)

app       = modal.App("editlens-squad-comparison", image=image)
hf_secret = modal.Secret.from_name("huggingface-secret")


@app.function(cpu=4, memory=8192, timeout=3600, secrets=[hf_secret])
def run_comparison(matched: list[dict], detector_bytes: bytes) -> dict:
    import io
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score
    from scipy.special import softmax
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer,
        DataCollatorWithPadding, Trainer, TrainingArguments,
    )

    # -----------------------------------------------------------------------
    # EditLens inference
    # -----------------------------------------------------------------------
    def run_editlens(texts):
        print("Loading EditLens...")
        tokenizer = AutoTokenizer.from_pretrained(EDITLENS_BASE_MODEL)
        model     = AutoModelForSequenceClassification.from_pretrained(EDITLENS_CHECKPOINT)
        model.eval()

        n_buckets = model.config.num_labels
        print(f"EditLens n_buckets={n_buckets}")

        def tokenize(example):
            return tokenizer(example["text"], truncation=True, max_length=512)

        ds_tok = Dataset.from_dict({"text": texts}).map(tokenize, num_proc=1)

        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir="/tmp/editlens_inf",
                per_device_eval_batch_size=16,
                report_to="none",
            ),
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
        )

        output = trainer.predict(ds_tok)
        probs  = softmax(output.predictions, axis=1)
        return ((probs @ np.arange(n_buckets)) / (n_buckets - 1)).tolist()

    # -----------------------------------------------------------------------
    # Our detector (encoder + classifier only)
    # -----------------------------------------------------------------------
    def run_ours(samples):
        feats = []
        for s in samples:
            cz = s["per_model_curvature_z"]
            feats.append([
                cz["mistral-7b"],
                cz["phi-3-mini"],
                cz["qwen-7b"],
                s["C_mean"],
                s["C_var"],
            ])

        x = torch.tensor(feats, dtype=torch.float32)

        class Detector(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder    = nn.Sequential(nn.Linear(5, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
                self.classifier = nn.Linear(32, 1)
            def forward(self, x):
                return self.classifier(self.encoder(x)).squeeze(-1)

        model = Detector()
        state = torch.load(io.BytesIO(detector_bytes), map_location="cpu", weights_only=True)
        model.load_state_dict(
            {k: v for k, v in state.items() if k.startswith("encoder") or k.startswith("classifier")},
            strict=True,
        )
        model.eval()
        with torch.no_grad():
            return torch.sigmoid(model(x)).tolist()

    texts  = [s["full_text"] for s in matched]
    labels = [s["label"] for s in matched]

    print(f"Running EditLens on {len(texts)} samples...")
    editlens_scores = run_editlens(texts)

    print("Running our detector...")
    our_scores = run_ours(matched)

    el_auroc  = roc_auc_score(labels, editlens_scores)
    our_auroc = roc_auc_score(labels, our_scores)

    print(f"\nEditLens AUROC : {el_auroc:.4f}")
    print(f"Ours AUROC     : {our_auroc:.4f}")

    return {
        "editlens_auroc": el_auroc,
        "our_auroc":      our_auroc,
        "n":              len(matched),
        "samples": [
            {**s, "editlens_score": editlens_scores[i], "our_score": our_scores[i]}
            for i, s in enumerate(matched)
        ],
    }


def find_optimal_threshold(scores, labels, num_thresholds=1000):
    import numpy as np
    best_t, best_f1 = 0.0, 0.0
    for t in np.linspace(0, 1, num_thresholds):
        preds = (np.array(scores) >= t).astype(int)
        tp = np.sum((preds == 1) & (np.array(labels) == 1))
        fp = np.sum((preds == 1) & (np.array(labels) == 0))
        fn = np.sum((preds == 0) & (np.array(labels) == 1))
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def threshold_eval(samples):
    import random
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score

    random.seed(42)
    shuffled = samples[:]
    random.shuffle(shuffled)
    split    = len(shuffled) // 2
    val, test = shuffled[:split], shuffled[split:]

    val_labels, test_labels = [s["label"] for s in val], [s["label"] for s in test]
    val_el,   test_el   = [s["editlens_score"] for s in val], [s["editlens_score"] for s in test]
    val_our,  test_our  = [s["our_score"] for s in val],      [s["our_score"] for s in test]

    el_thresh,  el_val_f1  = find_optimal_threshold(val_el,  val_labels)
    our_thresh, our_val_f1 = find_optimal_threshold(val_our, val_labels)

    def report(name, labels, scores, threshold):
        preds = (np.array(scores) >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        acc   = accuracy_score(labels, preds)
        f1    = f1_score(labels, preds, zero_division=0)
        auroc = roc_auc_score(labels, scores)
        fpr   = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr   = fn / (fn + tp) if (fn + tp) > 0 else 0
        print(f"\n  {name}  (threshold={threshold:.4f})")
        print(f"  {'':20} Pred Human   Pred AI")
        print(f"  {'True Human':20} {tn:>10}   {fp:>7}")
        print(f"  {'True AI':20} {fn:>10}   {tp:>7}")
        print(f"  Acc={acc:.3f}  F1={f1:.3f}  AUROC={auroc:.4f}  FPR={fpr:.3f}  FNR={fnr:.3f}")
        return {"threshold": threshold, "accuracy": acc, "f1": f1, "auroc": auroc,
                "fpr": fpr, "fnr": fnr, "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}

    print(f"\n{'='*60}")
    print(f"WIKIPEDIA/SQUAD  (val={len(val)}, test={len(test)})")
    print(f"  Val F1 — EditLens: {el_val_f1:.3f} @ {el_thresh:.4f} | "
          f"Ours: {our_val_f1:.3f} @ {our_thresh:.4f}")
    print(f"{'='*60}")

    return {
        "n_val":  len(val),
        "n_test": len(test),
        "editlens": {**report("EditLens RoBERTa", test_labels, test_el,  el_thresh), "val_f1": el_val_f1},
        "ours":     {**report("Ours            ", test_labels, test_our, our_thresh), "val_f1": our_val_f1},
    }


@app.local_entrypoint()
def main():
    # Load wikipedia samples from aggregated.json
    with open(AGG_PATH) as f:
        agg = json.load(f)
    wiki = [s for s in agg if s["domain"] == "wikipedia"]
    print(f"Wikipedia samples in aggregated.json: {len(wiki)}")

    # Build full-text lookup from raw + generated files
    print("Building text lookup...")
    lookup = {}
    for dataset in ["squad"]:
        with open(os.path.join(RAW_DIR, f"{dataset}.json")) as f:
            for s in json.load(f):
                lookup[s["text"][:100]] = s["text"]
        gen_dataset_dir = os.path.join(GEN_DIR, dataset)
        if not os.path.exists(gen_dataset_dir):
            continue
        for fname in os.listdir(gen_dataset_dir):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(gen_dataset_dir, fname)) as f:
                for s in json.load(f):
                    lookup[s["generated_text"][:100]] = s["generated_text"]

    # Only keep held-out generators (never seen during training) + human
    HELD_OUT_GENERATORS = {"gemma-7b", "phi-3-mini", "deepseek-7b"}
    wiki = [s for s in wiki if s["source_model"] in HELD_OUT_GENERATORS or s["source_model"] == "human"]
    print(f"After filtering to held-out generators + human: {len(wiki)}")

    # Match curvature samples to full texts
    matched, skipped = [], 0
    for s in wiki:
        full = lookup.get(s["text_prefix"])
        if full is None:
            skipped += 1
            continue
        matched.append({**s, "full_text": full})
    print(f"Matched {len(matched)}/{len(wiki)} samples ({skipped} skipped)")

    with open(DETECTOR_PT, "rb") as f:
        detector_bytes = f.read()

    print("Launching Modal job...")
    results = run_comparison.remote(matched, detector_bytes)

    scored_samples = results.pop("samples")

    print(f"\n{'='*60}")
    print(f"Wikipedia/SQuAD (n={results['n']})")
    print(f"  EditLens RoBERTa : AUROC = {results['editlens_auroc']:.4f}")
    print(f"  Ours             : AUROC = {results['our_auroc']:.4f}")

    results["threshold_eval"] = threshold_eval(scored_samples)

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")
