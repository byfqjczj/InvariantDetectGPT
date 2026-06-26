"""
Train the InvariantDetector on 17-dim probabilistic features.

Architecture: Linear(17→64) → ReLU → Linear(64→32) → ReLU → Linear(32→1)
Loss: L_cls + 0.1*L_adv + 0.1*L_ctr

Train/test split (both axes held out simultaneously):
  Training domains:    XSum, WritingPrompts
  Training generators: Mistral-7B, Qwen-7B, Gemma-7B
  Test domain:         SQuAD
  Test generators:     DeepSeek-7B, Phi-3-mini, LLaMA-2-13B
"""

import json
import os
import numpy as np
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.3.0", "scikit-learn", "numpy")
)

app = modal.App("ai-detector-training", image=image)


@app.function(cpu=4, memory=4096, timeout=3600)
def run_training(data: list[dict]) -> tuple:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from sklearn.metrics import roc_auc_score

    LAMBDA_ADV  = 0.1
    LAMBDA_CTR  = 0.1
    TEMPERATURE = 0.5
    EPOCHS      = 50
    LR          = 1e-3
    BATCH_SIZE  = 32
    HIDDEN_DIM  = 64
    REPR_DIM    = 32
    INPUT_DIM   = 17

    TRAIN_DOMAINS    = {"xsum", "writingprompts"}
    TEST_DOMAIN      = "squad"
    TRAIN_GENERATORS = {"mistral-7b", "qwen-7b", "gemma-7b"}
    TEST_GENERATORS  = {"deepseek-7b", "phi-3-mini", "llama2-13b"}

    FEATURE_KEYS   = ["curvature", "log_likelihood", "rank", "margin", "entropy"]
    MODELS_ORDERED = ["mistral-7b", "phi-3-mini", "qwen-7b"]

    def build_vector(s):
        vec = []
        for model in MODELS_ORDERED:
            for feat in FEATURE_KEYS:
                vec.append(s["per_model_features"][feat][model])
        vec.append(s["C_mean"])
        vec.append(s["C_var"])
        return vec  # 17-dim

    # Split along both domain and generator axes simultaneously
    train_data, test_data = [], []
    for s in data:
        src = s["source_model"]
        dom = s["domain"]
        if src == "human":
            if dom in TRAIN_DOMAINS:
                train_data.append(s)
            elif dom == TEST_DOMAIN:
                test_data.append(s)
        elif src in TRAIN_GENERATORS and dom in TRAIN_DOMAINS:
            train_data.append(s)
        elif src in TEST_GENERATORS and dom == TEST_DOMAIN:
            test_data.append(s)

    print(f"Train: {len(train_data)} | Test: {len(test_data)}")

    all_sources = sorted({s["source_model"] for s in data})
    all_domains = sorted({s["domain"] for s in data})
    src_to_idx  = {s: i for i, s in enumerate(all_sources)}
    dom_to_idx  = {d: i for i, d in enumerate(all_domains)}

    class FeatureDataset(Dataset):
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            s        = self.samples[idx]
            features = torch.tensor(build_vector(s), dtype=torch.float32)
            label    = torch.tensor(s["label"], dtype=torch.float32)
            src      = torch.tensor(src_to_idx[s["source_model"]], dtype=torch.long)
            dom      = torch.tensor(dom_to_idx[s["domain"]], dtype=torch.long)
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
            self.classifier    = nn.Linear(repr_dim, 1)
            self.src_adversary = nn.Linear(repr_dim, n_sources)
            self.dom_adversary = nn.Linear(repr_dim, n_domains)

        def forward(self, x, alpha=1.0):
            u     = self.encoder(x)
            y_hat = self.classifier(u).squeeze(-1)
            u_rev = grad_reverse(u, alpha)
            return u, y_hat, self.src_adversary(u_rev), self.dom_adversary(u_rev)

    def contrastive_loss(u, labels):
        u      = F.normalize(u, dim=-1)
        sim    = torch.matmul(u, u.T) / TEMPERATURE
        labels = labels.unsqueeze(1)
        same   = (labels == labels.T).float()
        mask   = 1 - torch.eye(u.size(0))
        same   = same * mask
        exp_sim  = torch.exp(sim) * mask
        log_prob = sim - torch.log(exp_sim.sum(dim=-1, keepdim=True) + 1e-8)
        loss     = -(log_prob * same).sum(dim=-1) / (same.sum(dim=-1) + 1e-8)
        return loss.mean()

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

    train_ds     = FeatureDataset(train_data)
    test_ds      = FeatureDataset(test_data)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    model     = InvariantDetector(INPUT_DIM, HIDDEN_DIM, REPR_DIM, len(all_sources), len(all_domains))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    results = {}
    for epoch in range(EPOCHS):
        model.train()
        alpha      = 2.0 / (1.0 + np.exp(-10 * epoch / EPOCHS)) - 1.0
        total_loss = 0
        for features, labels, src, dom in train_loader:
            optimizer.zero_grad()
            u, y_hat, src_logits, dom_logits = model(features, alpha=alpha)
            l_cls  = F.binary_cross_entropy_with_logits(y_hat, labels)
            l_adv  = F.cross_entropy(src_logits, src) + F.cross_entropy(dom_logits, dom)
            l_ctr  = contrastive_loss(u, labels)
            loss   = l_cls + LAMBDA_ADV * l_adv + LAMBDA_CTR * l_ctr
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            train_auroc = evaluate(model, train_loader)
            test_auroc  = evaluate(model, test_loader)
            print(f"Epoch {epoch+1:3d} | loss: {total_loss/len(train_loader):.4f} "
                  f"| train AUROC: {train_auroc:.4f} | test AUROC: {test_auroc:.4f}")
            results[epoch + 1] = {"train_auroc": train_auroc, "test_auroc": test_auroc}

    results["final"] = {
        "train_auroc": evaluate(model, train_loader),
        "test_auroc":  evaluate(model, test_loader),
    }

    import io
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    model_bytes = buf.getvalue()

    return results, model_bytes


@app.local_entrypoint()
def main():
    data_path = os.path.join(os.path.dirname(__file__), "..", "features", "extracted", "aggregated_17dim.json")
    with open(data_path) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples, running training on Modal...")
    results, model_bytes = run_training.remote(data)

    print("\n--- Results ---")
    for epoch, metrics in results.items():
        print(f"Epoch {epoch}: train AUROC={metrics['train_auroc']:.4f} | test AUROC={metrics['test_auroc']:.4f}")

    out_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")

    model_path = os.path.join(os.path.dirname(__file__), "detector.pt")
    with open(model_path, "wb") as f:
        f.write(model_bytes)
    print(f"Model saved to {model_path}")
