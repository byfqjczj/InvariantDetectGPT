"""
Generate AI text counterparts for each human sample using Modal GPU instances.

Training generators:  Mistral-7B, Qwen-7B  (LLaMA-3-8B pending HF approval)
Held-out generators:  Gemma-7B, Phi-3-mini, DeepSeek-7B

Uses @app.cls so each model is loaded ONCE per container and reused across
all batches/datasets — 5 containers total instead of thousands.

Output: generation/generated/<dataset>/<model_key>.json
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
    )
)

app = modal.App("ai-text-generation", image=image)
hf_secret = modal.Secret.from_name("huggingface-secret")

GENERATORS = {
    "mistral-7b":  "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen-7b":     "Qwen/Qwen2-7B-Instruct",
    "gemma-7b":    "google/gemma-7b-it",
    "phi-3-mini":  "microsoft/Phi-3-mini-4k-instruct",
    "deepseek-7b": "deepseek-ai/deepseek-llm-7b-chat",
}

DATASETS = ["xsum", "writingprompts", "squad"]


# ---------------------------------------------------------------------------
# One class per model — model loaded once in enter(), reused for all batches
# ---------------------------------------------------------------------------
@app.cls(
    gpu="A10G",
    timeout=86400,  # 24h — enough for all datasets
    secrets=[hf_secret],
    max_containers=5,
)
class Generator:
    model_key: str = modal.parameter()

    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        model_id = GENERATORS[self.model_key]
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
    def generate(self, samples: list[dict]) -> list[dict]:
        import torch

        results = []
        for i, sample in enumerate(samples):
            prompt = sample["prompt"]
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=256
            ).to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            results.append({
                "prompt": prompt,
                "generated_text": generated_text,
                "model": self.model_key,
                "domain": sample["domain"],
            })
            print(f"[{self.model_key}] {i+1}/{len(samples)} done")

        return results


# ---------------------------------------------------------------------------
# Local entrypoint — spins up one Generator per model, processes all datasets
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    raw_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    gen_dir = os.path.join(os.path.dirname(__file__), "generated")

    # Build per-model sample lists
    pending = {}  # model_key -> (all_samples, pending_datasets)
    for model_key in GENERATORS:
        all_samples = []
        pending_datasets = []
        for dataset in DATASETS:
            out_path = os.path.join(gen_dir, dataset, f"{model_key}.json")
            if os.path.exists(out_path):
                print(f"Skipping {model_key} x {dataset} (already done)")
                continue
            with open(os.path.join(raw_dir, f"{dataset}.json")) as f:
                samples = json.load(f)[:100]
            all_samples.extend(samples)
            pending_datasets.append(dataset)

        if all_samples:
            pending[model_key] = (all_samples, pending_datasets)

    if not pending:
        print("All jobs already complete.")
        return

    # Fan out all models in parallel — each gets its own container
    print(f"Launching {len(pending)} model containers in parallel...")
    futures = {
        model_key: Generator(model_key=model_key).generate.spawn(all_samples)
        for model_key, (all_samples, _) in pending.items()
    }

    # Collect results as they complete and save
    for model_key, future in futures.items():
        _, pending_datasets = pending[model_key]
        results = future.get()
        print(f"[{model_key}] Got {len(results)} results, saving...")

        idx = 0
        for dataset in pending_datasets:
            n = 100
            dataset_results = results[idx:idx + n]
            idx += n

            out_dir = os.path.join(gen_dir, dataset)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{model_key}.json")
            with open(out_path, "w") as f:
                json.dump(dataset_results, f, indent=2)
            print(f"Saved {len(dataset_results)} samples -> {out_path}")
