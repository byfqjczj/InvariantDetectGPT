"""
Modal version of compare_editlens.py.

Runs EditLens RoBERTa + our InvariantDetector on all samples and reports
AUROC side-by-side: overall, by domain, by source model, train vs held-out.

Usage:
  cd implementation
  PYTHONUTF8=1 modal run evaluation/compare_editlens_modal.py

Output:
  evaluation/comparison_results.json
"""

import json
import os
import modal

# ---------------------------------------------------------------------------
# Paths (local)
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR     = os.path.join(BASE_DIR, "data", "raw")
GEN_DIR     = os.path.join(BASE_DIR, "generation", "generated")
AGG_PATH    = os.path.join(BASE_DIR, "features", "extracted", "aggregated.json")
DETECTOR_PT = os.path.join(BASE_DIR, "training", "detector.pt")
OUT_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comparison_results.json")

EDITLENS_SCRIPTS = os.path.abspath(os.path.join(BASE_DIR, "..", "editlens", "scripts"))

DATASETS         = ["xsum", "writingprompts", "squad"]
TRAIN_GENERATORS = {"mistral-7b", "qwen-7b"}
HELD_GENERATORS  = {"gemma-7b", "phi-3-mini", "deepseek-7b"}

EDITLENS_CHECKPOINT = "pangram/editlens_roberta-large"
EDITLENS_BASE_MODEL = "FacebookAI/roberta-large"

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------
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
        "emoji",
        "numpy",
        "huggingface_hub",
        "accelerate",
    )
)

app = modal.App("editlens-comparison", image=image)
hf_secret = modal.Secret.from_name("huggingface-secret")

# ---------------------------------------------------------------------------
# Modal function
# ---------------------------------------------------------------------------
@app.function(
    cpu=4,
    memory=8192,
    timeout=3600,
    secrets=[hf_secret],
)
def run_comparison(
    matched: list[dict],
    detector_bytes: bytes,
) -> dict:
    import io
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score
    from datasets import Dataset
    from scipy.special import softmax
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    import re
    import emoji

    def clean_text(text):
        text = emoji.demojize(text)
        if "</think>" in text:
            text = text.split("</think>")[1].strip()
        paragraphs = [p for p in text.split("\n") if p.strip()]
        if paragraphs:
            first = re.sub(r"^[^a-zA-Z0-9]*", "", paragraphs[0])
            first = emoji.replace_emoji(first, "")
            if any(first.startswith(p) for p in ["Sure", "Here", "Abstract", "Title", "I'm happy to help", "Certainly"]):
                if len(paragraphs) > 1:
                    text = "\n".join(paragraphs[1:])
        text = text.lower()
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # -----------------------------------------------------------------------
    # EditLens inference
    # -----------------------------------------------------------------------
    def run_editlens(texts):
        print("Loading EditLens tokenizer + model...")
        tokenizer = AutoTokenizer.from_pretrained(EDITLENS_BASE_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(EDITLENS_CHECKPOINT)
        model.eval()

        n_buckets = model.config.num_labels
        print(f"EditLens n_buckets={n_buckets}")

        def tokenize(example):
            return tokenizer(
                clean_text(example["text"]),
                truncation=True,
                max_length=512,
            )

        ds = Dataset.from_dict({"text": texts})
        ds_tok = ds.map(tokenize, num_proc=1)

        training_args = TrainingArguments(
            output_dir="/tmp/editlens_inf",
            per_device_eval_batch_size=16,
            report_to="none",
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
        )

        output = trainer.predict(ds_tok)
        probs = softmax(output.predictions, axis=1)
        bucket_labels = np.arange(n_buckets)
        scores = (probs @ bucket_labels) / (n_buckets - 1)
        return scores.tolist()

    # -----------------------------------------------------------------------
    # Our InvariantDetector inference
    # -----------------------------------------------------------------------
    class GradientReversal(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, alpha):
            ctx.alpha = alpha
            return x

        @staticmethod
        def backward(ctx, grad):
            return -ctx.alpha * grad, None

    class InvariantDetector(nn.Module):
        def __init__(self, input_dim, hidden_dim, repr_dim, n_sources, n_domains):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, repr_dim),
                nn.ReLU(),
            )
            self.classifier = nn.Linear(repr_dim, 1)
            self.src_adversary = nn.Linear(repr_dim, n_sources)
            self.dom_adversary = nn.Linear(repr_dim, n_domains)

        def forward(self, x, alpha=1.0):
            u = self.encoder(x)
            y_hat = self.classifier(u).squeeze(-1)
            u_rev = GradientReversal.apply(u, alpha)
            return u, y_hat, self.src_adversary(u_rev), self.dom_adversary(u_rev)

    def run_ours(samples):
        all_sources = sorted({s["source_model"] for s in samples})
        all_domains = sorted({s["domain"] for s in samples})

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
        model = InvariantDetector(5, 64, 32, len(all_sources), len(all_domains))
        buf = io.BytesIO(detector_bytes)
        state = torch.load(buf, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()

        with torch.no_grad():
            _, y_hat, _, _ = model(x)
            return torch.sigmoid(y_hat).tolist()

    # -----------------------------------------------------------------------
    # Run both
    # -----------------------------------------------------------------------
    texts  = [s["full_text"] for s in matched]
    labels = [s["label"] for s in matched]

    print(f"Running EditLens on {len(texts)} samples...")
    editlens_scores = run_editlens(texts)

    print("Running our detector...")
    our_scores = run_ours(matched)

    # -----------------------------------------------------------------------
    # AUROC reporting
    # -----------------------------------------------------------------------
    def auroc(labs, scores):
        labs, scores = list(labs), list(scores)
        if len(set(labs)) < 2:
            return float("nan")
        return roc_auc_score(labs, scores)

    def make_result(labs, el_scores, ou_scores):
        return {
            "n": len(labs),
            "editlens_auroc": auroc(labs, el_scores),
            "ours_auroc":     auroc(labs, ou_scores),
        }

    results = {}
    results["overall"] = make_result(labels, editlens_scores, our_scores)

    for domain in sorted({s["domain"] for s in matched}):
        idx = [i for i, s in enumerate(matched) if s["domain"] == domain]
        results[f"domain_{domain}"] = make_result(
            [labels[i] for i in idx],
            [editlens_scores[i] for i in idx],
            [our_scores[i] for i in idx],
        )

    for group, gen_set in [("train_generators", TRAIN_GENERATORS), ("held_out_generators", HELD_GENERATORS)]:
        idx = [i for i, s in enumerate(matched) if s["source_model"] in gen_set]
        if idx:
            results[group] = make_result(
                [labels[i] for i in idx],
                [editlens_scores[i] for i in idx],
                [our_scores[i] for i in idx],
            )

    for src in sorted({s["source_model"] for s in matched if s["source_model"] != "human"}):
        idx = [i for i, s in enumerate(matched) if s["source_model"] in (src, "human")]
        results[f"src_{src}"] = make_result(
            [labels[i] for i in idx],
            [editlens_scores[i] for i in idx],
            [our_scores[i] for i in idx],
        )

    # Attach per-sample scores for saving locally
    results["_samples"] = [
        {**s, "editlens_score": editlens_scores[i], "our_score": our_scores[i]}
        for i, s in enumerate(matched)
    ]

    return results


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    # Build full-text lookup
    print("Building text lookup...")
    lookup = {}
    for dataset in DATASETS:
        with open(os.path.join(RAW_DIR, f"{dataset}.json")) as f:
            for s in json.load(f):
                lookup[s["text"][:100]] = s["text"]
        dataset_gen_dir = os.path.join(GEN_DIR, dataset)
        if not os.path.exists(dataset_gen_dir):
            continue
        for fname in os.listdir(dataset_gen_dir):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(dataset_gen_dir, fname)) as f:
                for s in json.load(f):
                    lookup[s["generated_text"][:100]] = s["generated_text"]

    # Match aggregated samples to full texts
    with open(AGG_PATH) as f:
        agg = json.load(f)

    matched, skipped = [], 0
    for s in agg:
        full = lookup.get(s["text_prefix"])
        if full is None:
            skipped += 1
            continue
        matched.append({**s, "full_text": full})

    print(f"Matched {len(matched)} / {len(agg)} samples ({skipped} skipped)")

    # Load detector weights
    with open(DETECTOR_PT, "rb") as f:
        detector_bytes = f.read()

    # Run on Modal
    print("Launching Modal job...")
    results = run_comparison.remote(matched, detector_bytes)

    # Print summary
    samples = results.pop("_samples")
    print("\n" + "=" * 70)
    print("AUROC COMPARISON: EditLens RoBERTa  vs  InvariantDetector (Ours)")
    print("=" * 70)
    for key, val in results.items():
        label = key.replace("_", " ").replace("domain ", "Domain: ").replace("src ", "human vs ")
        print(f"  {label:<35} n={val['n']:>4}  EditLens={val['editlens_auroc']:.4f}  Ours={val['ours_auroc']:.4f}")
    print("=" * 70)

    results["samples"] = samples
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")
