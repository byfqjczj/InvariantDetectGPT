"""
Extract 17-dim feature vectors from the original full-length dataset.

Loads directly from data/raw/ and generation/generated/ to avoid the 200-char
truncation introduced by extract_features.py storing only text[:200].

Input:  data/raw/{xsum,writingprompts,squad}.json  +  generation/generated/**/*.json
Output: features/extracted/aggregated_17dim.json

Feature vector layout (17 dims):
  [mistral curvature_z, mistral log_likelihood_z, mistral rank_z, mistral margin_z, mistral entropy_z,
   phi     curvature_z, phi     log_likelihood_z, phi     rank_z, phi     margin_z, phi     entropy_z,
   qwen    curvature_z, qwen    log_likelihood_z, qwen    rank_z, qwen    margin_z, qwen    entropy_z,
   C_mean, C_var]

Usage:
  cd implementation
  modal run features/extract_17dim.py
"""

import json
import os
import modal

REFERENCE_MODELS = {
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "phi-3-mini":  "microsoft/Phi-3-mini-4k-instruct",
    "qwen-7b":     "Qwen/Qwen2-7B-Instruct",
}
N_ALTERNATIVES = 20
FEATURE_KEYS   = ["curvature", "log_likelihood", "rank", "margin", "entropy"]
MODELS_ORDERED = ["mistral-7b", "phi-3-mini", "qwen-7b"]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEATS_DIR  = os.path.join(SCRIPT_DIR, "extracted")
OUT_PATH   = os.path.join(FEATS_DIR, "aggregated_17dim.json")
RAW_DIR    = os.path.join(SCRIPT_DIR, "..", "data", "raw")
GEN_DIR    = os.path.join(SCRIPT_DIR, "..", "generation", "generated")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.0",
        "transformers==4.44.0",
        "accelerate==0.33.0",
        "huggingface_hub==0.24.0",
        "bitsandbytes==0.43.1",
        "sentencepiece",
        "protobuf",
        "numpy",
        "scikit-learn",
    )
)

app       = modal.App("extract-17dim-features", image=image)
hf_secret = modal.Secret.from_name("huggingface-secret")


@app.cls(gpu="A10G", timeout=86400, secrets=[hf_secret], memory=16384)
class CurvatureExtractor:
    model_key: str = modal.parameter()

    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_id       = REFERENCE_MODELS[self.model_key]
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model     = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_cfg, device_map="auto", trust_remote_code=True
        )
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"[{self.model_key}] loaded")

    @modal.method()
    def extract(self, samples: list[dict]) -> list[dict]:
        import torch
        import numpy as np

        results = []
        for i, sample in enumerate(samples):
            text   = sample["text"]
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            ).to(self.model.device)
            ids = inputs["input_ids"][0]
            T   = len(ids)
            if T < 2:
                continue

            with torch.no_grad():
                logits = self.model(**inputs).logits[0]

            log_probs       = torch.nn.functional.log_softmax(logits[:-1], dim=-1)
            probs           = torch.exp(log_probs)
            token_log_probs = log_probs[range(T - 1), ids[1:]]

            curvatures, log_likelihoods, ranks, margins, entropies = [], [], [], [], []
            for t in range(T - 1):
                lp_t   = token_log_probs[t]
                lp_row = log_probs[t]
                p_row  = probs[t]

                alt_idx = torch.multinomial(p_row, num_samples=N_ALTERNATIVES, replacement=True)
                curvatures.append(lp_t.item() - lp_row[alt_idx].mean().item())
                log_likelihoods.append(lp_t.item())
                ranks.append((lp_row > lp_t).float().mean().item())
                margins.append((lp_row.max() - lp_t).item())
                entropies.append(-(p_row * lp_row).sum().item())

            results.append({
                "text_prefix":    text[:100],
                "label":          sample["label"],
                "source_model":   sample["source_model"],
                "domain":         sample["domain"],
                "curvature":      float(np.mean(curvatures)),
                "log_likelihood": float(np.mean(log_likelihoods)),
                "rank":           float(np.mean(ranks)),
                "margin":         float(np.mean(margins)),
                "entropy":        float(np.mean(entropies)),
            })

            if (i + 1) % 20 == 0:
                print(f"[{self.model_key}] {i+1}/{len(samples)}")

        return results


def extract_raw(samples: list[dict], n_chunks: int = 9) -> dict:
    n = len(samples)
    base, rem = divmod(n, n_chunks)
    chunks, offset = [], 0
    for i in range(n_chunks):
        size = base + (1 if i < rem else 0)
        chunks.append(samples[offset:offset + size])
        offset += size

    print(f"Spawning {n_chunks * len(REFERENCE_MODELS)} containers "
          f"({n_chunks} chunks × {len(REFERENCE_MODELS)} models, ~{base} samples/chunk)...")

    futures = {}
    for key in REFERENCE_MODELS:
        extractor = CurvatureExtractor(model_key=key)
        for i, chunk in enumerate(chunks):
            futures[(key, i)] = extractor.extract.spawn(chunk)

    ref_data = {key: [] for key in REFERENCE_MODELS}
    for (key, i), fut in futures.items():
        rows = fut.get()
        ref_data[key].extend(rows)
        print(f"  [{key}] chunk {i}: {len(rows)} rows")

    for key, rows in ref_data.items():
        print(f"  [{key}] total: {len(rows)} rows")

    return ref_data


def aggregate_features(ref_data: dict) -> list[dict]:
    import numpy as np
    from collections import defaultdict

    # Z-score each feature per model, fitted on this dataset
    for model_key, rows in ref_data.items():
        for feat in FEATURE_KEYS:
            vals  = np.array([r[feat] for r in rows])
            mu, sigma = vals.mean(), vals.std() + 1e-8
            for r in rows:
                r[f"{feat}_z"] = (r[feat] - mu) / sigma

    # Group by text_prefix to align across models
    grouped = defaultdict(dict)
    for model_key, rows in ref_data.items():
        for r in rows:
            key = (r["text_prefix"], r["label"], r["source_model"], r["domain"])
            grouped[key][model_key] = {f: r[f"{f}_z"] for f in FEATURE_KEYS}

    aggregated = []
    for (text_prefix, label, source_model, domain), by_model in grouped.items():
        if len(by_model) < len(REFERENCE_MODELS):
            continue  # skip samples missing any model
        curvature_z_vals = np.array([by_model[m]["curvature"] for m in MODELS_ORDERED])
        per_model = {}
        for m in MODELS_ORDERED:
            for f in FEATURE_KEYS:
                per_model.setdefault(f, {})[m] = float(by_model[m][f])
        aggregated.append({
            "text_prefix":        text_prefix,
            "label":              label,
            "source_model":       source_model,
            "domain":             domain,
            "C_mean":             float(curvature_z_vals.mean()),
            "C_var":              float(curvature_z_vals.var()),
            "per_model_features": per_model,  # {feat: {model: z_val}}
        })

    return aggregated


def load_samples() -> list[dict]:
    """Load full-length texts from original raw + generated sources."""
    DATASETS = ["xsum", "writingprompts", "squad"]
    HUMAN_PER_DATASET = 100
    AI_PER_MODEL_PER_DATASET = 34  # 3 generators per split × 34 ≈ 100 human per domain → ~50/50

    samples = []
    for dataset in DATASETS:
        raw_path = os.path.join(RAW_DIR, f"{dataset}.json")
        with open(raw_path) as f:
            human_rows = json.load(f)[:HUMAN_PER_DATASET]
        for r in human_rows:
            samples.append({
                "text":         r["text"],
                "label":        0,
                "source_model": "human",
                "domain":       dataset,
            })

        dataset_gen_dir = os.path.join(GEN_DIR, dataset)
        if not os.path.exists(dataset_gen_dir):
            continue
        for fname in sorted(os.listdir(dataset_gen_dir)):
            if not fname.endswith(".json"):
                continue
            model_name = fname.replace(".json", "")
            with open(os.path.join(dataset_gen_dir, fname)) as f:
                ai_rows = json.load(f)[:AI_PER_MODEL_PER_DATASET]
            for r in ai_rows:
                text = r.get("generated_text") or r.get("text", "")
                samples.append({
                    "text":         text,
                    "label":        1,
                    "source_model": model_name,
                    "domain":       dataset,
                })

    return samples


@app.local_entrypoint()
def main():
    samples = load_samples()
    print(f"Loaded {len(samples)} samples from original sources")

    ref_data   = extract_raw(samples)
    aggregated = aggregate_features(ref_data)

    print(f"\nAggregated {len(aggregated)} samples with 17-dim features")

    with open(OUT_PATH, "w") as f:
        json.dump(aggregated, f)

    print(f"Saved to {OUT_PATH}")

    # Quick sanity check
    s = aggregated[0]
    vec_len = sum(len(s["per_model_features"][feat]) for feat in FEATURE_KEYS) + 2
    print(f"Feature vector length: {vec_len} (expected 17)")