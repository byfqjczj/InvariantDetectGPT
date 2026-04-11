"""
Extract token-level probabilistic features for each text sample.

For each reference model M_k and each text x, we compute:
  - log_likelihood: mean token log-prob under M_k
  - token_rank:     mean rank of each actual token in M_k's distribution
  - entropy:        mean entropy of M_k's distribution at each position
  - curvature:      FastDetectGPT signal — log p(x) minus expected log p
                    of conditionally-sampled alternatives at each position

Uses @app.cls so each reference model loads once and processes all samples.
3 containers total (one per reference model), running in parallel.

Output: features/extracted/<ref_model>.json + features/extracted/aggregated.json
"""

import json
import os
import modal

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
    )
)

app = modal.App("feature-extraction", image=image)
hf_secret = modal.Secret.from_name("huggingface-secret")

REFERENCE_MODELS = {
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen-7b":    "Qwen/Qwen2-7B-Instruct",
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
}

N_ALTERNATIVES = 20  # alternatives sampled per token for curvature


@app.cls(
    gpu="A10G",
    timeout=86400,
    secrets=[hf_secret],
    max_containers=3,
)
class FeatureExtractor:
    model_key: str = modal.parameter()

    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        model_id = REFERENCE_MODELS[self.model_key]
        print(f"[{self.model_key}] Loading {model_id}...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[{self.model_key}] Model loaded.")

    @modal.method()
    def extract(self, samples: list[dict]) -> list[dict]:
        import torch
        import numpy as np

        results = []
        for i, sample in enumerate(samples):
            text = sample.get("generated_text") or sample.get("text", "")
            if not text.strip():
                continue

            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            ).to(self.model.device)
            input_ids = inputs["input_ids"][0]
            T = len(input_ids)

            if T < 2:
                continue

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0]  # (T, V)

            shift_logits = logits[:-1]       # (T-1, V)
            shift_labels = input_ids[1:]     # (T-1,)

            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            probs = torch.exp(log_probs)

            # log-likelihood
            token_log_probs = log_probs[range(T - 1), shift_labels]
            log_likelihood = token_log_probs.mean().item()

            # token rank
            sorted_indices = torch.argsort(shift_logits, dim=-1, descending=True)
            ranks = (sorted_indices == shift_labels.unsqueeze(1)).nonzero(as_tuple=False)[:, 1]
            token_rank = ranks.float().mean().item()

            # entropy
            entropy = -(probs * log_probs).sum(dim=-1).mean().item()

            # curvature (FastDetectGPT)
            position_curvatures = []
            for t in range(T - 1):
                actual_log_p = token_log_probs[t].item()
                alt_indices = torch.multinomial(probs[t], num_samples=N_ALTERNATIVES, replacement=True)
                alt_log_probs = log_probs[t][alt_indices]
                expected_alt_log_p = alt_log_probs.mean().item()
                position_curvatures.append(actual_log_p - expected_alt_log_p)

            curvature = float(np.mean(position_curvatures))

            results.append({
                "text": text[:200],  # store prefix only to save space
                "label": sample["label"],
                "domain": sample["domain"],
                "source_model": sample.get("model", "human"),
                "reference_model": self.model_key,
                "log_likelihood": log_likelihood,
                "token_rank": token_rank,
                "entropy": entropy,
                "curvature": curvature,
            })
            print(f"[{self.model_key}] {i+1}/{len(samples)} done")

        return results


@app.local_entrypoint()
def main():
    raw_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    gen_dir = os.path.join(os.path.dirname(__file__), "..", "generation", "generated")
    out_dir = os.path.join(os.path.dirname(__file__), "extracted")
    os.makedirs(out_dir, exist_ok=True)

    datasets = ["xsum", "writingprompts", "squad"]

    # Build combined human + AI sample list — 100 human and 100 AI per dataset
    all_samples = []
    for dataset in datasets:
        with open(os.path.join(raw_dir, f"{dataset}.json")) as f:
            human_samples = json.load(f)[:100]
        for s in human_samples:
            all_samples.append({**s, "label": 0, "model": "human"})

        # Collect all AI samples then subsample to 100 total
        dataset_gen_dir = os.path.join(gen_dir, dataset)
        if not os.path.exists(dataset_gen_dir):
            continue
        ai_samples = []
        for fname in os.listdir(dataset_gen_dir):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(dataset_gen_dir, fname)) as f:
                ai_samples.extend(json.load(f))

        # 100 total AI samples, spread evenly across models (20 per model)
        per_model = 100 // len(os.listdir(dataset_gen_dir))
        capped_ai = []
        by_model: dict[str, list] = {}
        for s in ai_samples:
            by_model.setdefault(s["model"], []).append(s)
        for model_samples in by_model.values():
            capped_ai.extend(model_samples[:per_model])

        for s in capped_ai[:100]:
            all_samples.append({**s, "label": 1})

    print(f"Total samples: {len(all_samples)}")

    # Fan out — one container per reference model, all in parallel
    futures = {
        model_key: FeatureExtractor(model_key=model_key).extract.spawn(all_samples)
        for model_key in REFERENCE_MODELS
        if not os.path.exists(os.path.join(out_dir, f"{model_key}.json"))
    }

    if not futures:
        print("All feature extraction already done, running aggregation...")
        _aggregate(out_dir)
        return

    print(f"Launched {len(futures)} reference model containers in parallel...")

    for model_key, future in futures.items():
        results = future.get()
        out_path = os.path.join(out_dir, f"{model_key}.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[{model_key}] Saved {len(results)} rows -> {out_path}")

    _aggregate(out_dir)


def _aggregate(out_dir: str):
    import numpy as np

    ref_data = {}
    for fname in os.listdir(out_dir):
        if fname.endswith(".json") and fname != "aggregated.json":
            ref_model = fname.replace(".json", "")
            with open(os.path.join(out_dir, fname)) as f:
                ref_data[ref_model] = json.load(f)

    if not ref_data:
        return

    # Z-score normalize curvature per reference model
    for ref_model, rows in ref_data.items():
        curvatures = np.array([r["curvature"] for r in rows])
        mu, sigma = curvatures.mean(), curvatures.std() + 1e-8
        for r in rows:
            r["curvature_z"] = (r["curvature"] - mu) / sigma

    # Group by text prefix and aggregate C_mean, C_var across models
    from collections import defaultdict
    grouped = defaultdict(dict)
    for ref_model, rows in ref_data.items():
        for r in rows:
            key = (r["text"][:100], r["label"], r["domain"], r["source_model"])
            grouped[key][ref_model] = r["curvature_z"]

    ref_models = list(ref_data.keys())
    aggregated = []
    for (text_prefix, label, domain, source_model), model_curvatures in grouped.items():
        if len(model_curvatures) < len(ref_models):
            continue
        z_vals = np.array([model_curvatures[m] for m in ref_models])
        aggregated.append({
            "text_prefix": text_prefix,
            "label": label,
            "domain": domain,
            "source_model": source_model,
            "C_mean": float(z_vals.mean()),
            "C_var": float(z_vals.var()),
            "per_model_curvature_z": {m: float(v) for m, v in zip(ref_models, z_vals)},
        })

    out_path = os.path.join(out_dir, "aggregated.json")
    with open(out_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"Aggregated {len(aggregated)} samples -> {out_path}")
