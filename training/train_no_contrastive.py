"""
Ablation: InvariantDetector trained WITHOUT contrastive loss.

Loss:
  L = L_cls + lambda_adv * L_adv

  L_cls: binary cross-entropy (AI vs human)
  L_adv: adversarial loss — gradient reversal on source model + domain heads

Train/held-out split at the generator level:
  Training generators:  mistral-7b, qwen-7b
  Held-out generators:  gemma-7b, phi-3-mini, deepseek-7b
"""

import json
import os
import numpy as np
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.3.0", "scikit-learn", "numpy")
)

app = modal.App("ai-detector-no-contrastive", image=image)


@app.function(cpu=4, memory=4096, timeout=3600)
def run_training(data: list[dict]) -> dict:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from sklearn.metrics import roc_auc_score

    LAMBDA_ADV  = 0.1
    EPOCHS      = 50
    LR          = 1e-3
    BATCH_SIZE  = 32
    HIDDEN_DIM  = 64
    REPR_DIM    = 32
    TRAIN_GENERATORS   = {"mistral-7b", "qwen-7b"}
    HELD_OUT_GENERATORS = {"gemma-7b", "phi-3-mini", "deepseek-7b"}

    import random
    random.seed(42)

    human_samples = [s for s in data if s["source_model"] == "human"]
    random.shuffle(human_samples)
    split = int(0.7 * len(human_samples))
    human_train    = human_samples[:split]
    human_held_out = human_samples[split:]

    train_data, held_out_data = list(human_train), list(human_held_out)
    for s in data:
        src = s["source_model"]
        if src in TRAIN_GENERATORS:
            train_data.append(s)
        elif src in HELD_OUT_GENERATORS:
            held_out_data.append(s)

    print(f"Train: {len(train_data)} | Held-out: {len(held_out_data)}")

    all_sources = sorted({s["source_model"] for s in data})
    all_domains = sorted({s["domain"] for s in data})
    src_to_idx  = {s: i for i, s in enumerate(all_sources)}
    dom_to_idx  = {d: i for i, d in enumerate(all_domains)}

    class CurvatureDataset(Dataset):
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            s = self.samples[idx]
            features = torch.tensor([
                s["per_model_curvature_z"]["mistral-7b"],
                s["per_model_curvature_z"]["phi-3-mini"],
                s["per_model_curvature_z"]["qwen-7b"],
                s["C_mean"],
                s["C_var"],
            ], dtype=torch.float32)
            label = torch.tensor(s["label"], dtype=torch.float32)
            src   = torch.tensor(src_to_idx[s["source_model"]], dtype=torch.long)
            dom   = torch.tensor(dom_to_idx[s["domain"]], dtype=torch.long)
            return features, label, src, dom

    class GradientReversal(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, alpha):
            ctx.alpha = alpha
            return x

        @staticmethod
        def backward(ctx, grad):
            return -ctx.alpha * grad, None

    def grad_reverse(x, alpha=1.0):
        return GradientReversal.apply(x, alpha)

    class InvariantDetector(nn.Module):
        def __init__(self, input_dim, hidden_dim, repr_dim, n_sources, n_domains):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, repr_dim),
                nn.ReLU(),
            )
            self.classifier  = nn.Linear(repr_dim, 1)
            self.src_adversary = nn.Linear(repr_dim, n_sources)
            self.dom_adversary = nn.Linear(repr_dim, n_domains)

        def forward(self, x, alpha=1.0):
            u     = self.encoder(x)
            y_hat = self.classifier(u).squeeze(-1)
            u_rev = grad_reverse(u, alpha)
            return u, y_hat, self.src_adversary(u_rev), self.dom_adversary(u_rev)

    def evaluate(model, loader):
        model.eval()
        preds, labs = [], []
        with torch.no_grad():
            for features, labels, _, _ in loader:
                _, y_hat, _, _ = model(features)
                preds.extend(torch.sigmoid(y_hat).tolist())
                labs.extend(labels.tolist())
        if len(set(labs)) < 2:
            return float("nan")
        return roc_auc_score(labs, preds)

    train_ds      = CurvatureDataset(train_data)
    held_out_ds   = CurvatureDataset(held_out_data)
    train_loader  = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    held_out_loader = DataLoader(held_out_ds, batch_size=BATCH_SIZE)

    model     = InvariantDetector(5, HIDDEN_DIM, REPR_DIM, len(all_sources), len(all_domains))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    results = {}
    for epoch in range(EPOCHS):
        model.train()
        alpha = 2.0 / (1.0 + np.exp(-10 * epoch / EPOCHS)) - 1.0
        total_loss = 0
        for features, labels, src, dom in train_loader:
            optimizer.zero_grad()
            u, y_hat, src_logits, dom_logits = model(features, alpha=alpha)
            l_cls  = F.binary_cross_entropy_with_logits(y_hat, labels)
            l_adv  = F.cross_entropy(src_logits, src) + F.cross_entropy(dom_logits, dom)
            loss   = l_cls + LAMBDA_ADV * l_adv
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            train_auroc = evaluate(model, train_loader)
            held_auroc  = evaluate(model, held_out_loader)
            print(f"Epoch {epoch+1:3d} | loss: {total_loss/len(train_loader):.4f} "
                  f"| train AUROC: {train_auroc:.4f} | held-out AUROC: {held_auroc:.4f}")
            results[epoch + 1] = {"train_auroc": train_auroc, "held_out_auroc": held_auroc}

    results["final"] = {
        "train_auroc": evaluate(model, train_loader),
        "held_out_auroc": evaluate(model, held_out_loader),
    }

    import io
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    model_bytes = buf.getvalue()

    return results, model_bytes


@app.local_entrypoint()
def main():
    data_path = os.path.join(os.path.dirname(__file__), "..", "features", "extracted", "aggregated.json")
    with open(data_path) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples, running training on Modal...")
    results, model_bytes = run_training.remote(data)

    print("\n--- Results ---")
    for epoch, metrics in results.items():
        print(f"Epoch {epoch}: train AUROC={metrics['train_auroc']:.4f} | held-out AUROC={metrics['held_out_auroc']:.4f}")

    out_path = os.path.join(os.path.dirname(__file__), "results_no_contrastive.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")

    model_path = os.path.join(os.path.dirname(__file__), "detector_no_contrastive.pt")
    with open(model_path, "wb") as f:
        f.write(model_bytes)
    print(f"Model saved to {model_path}")

    final = results["final"]
    epoch_rows = "\n".join(
        f"| {e} | {m['train_auroc']:.4f} | {m['held_out_auroc']:.4f} |"
        for e, m in results.items() if e != "final"
    )
    writeup = f"""# Detector Training Results (No Contrastive Loss)

## Model
- Architecture: 5 input features → Linear(64) → ReLU → Linear(32) → ReLU → Linear(1)
- Loss: L_cls (BCE) + 0.1 * L_adv (gradient reversal on source + domain)
- Optimizer: Adam, lr=1e-3, 50 epochs

## Data Split
- Training generators (seen): Mistral-7B, Qwen-7B
- Held-out generators (unseen): Gemma-7B, Phi-3-mini, DeepSeek-7B
- Human samples: 70% train / 30% held-out

## Results by Epoch

| Epoch | Train AUROC | Held-out AUROC |
|-------|-------------|----------------|
{epoch_rows}

## Final Performance
- **Train AUROC: {final['train_auroc']:.4f}**
- **Held-out AUROC: {final['held_out_auroc']:.4f}**
"""
    writeup_path = os.path.join(os.path.dirname(__file__), "writeup_no_contrastive.md")
    with open(writeup_path, "w") as f:
        f.write(writeup)
    print(f"Writeup saved to {writeup_path}")
