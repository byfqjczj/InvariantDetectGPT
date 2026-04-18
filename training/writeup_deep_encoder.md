# Detector Training Results

## Model
- Architecture: 5 input features → Linear(64) → ReLU → Linear(32) → ReLU → Linear(1)
- Loss: L_cls (BCE) + 0.1 * L_adv (gradient reversal on source + domain) + 0.1 * L_ctr (contrastive)
- Optimizer: Adam, lr=1e-3, 50 epochs

## Data Split
- Training generators (seen): Mistral-7B, Qwen-7B
- Held-out generators (unseen): Gemma-7B, Phi-3-mini, DeepSeek-7B
- Human samples: 70% train / 30% held-out

## Results by Epoch

| Epoch | Train AUROC | Held-out AUROC |
|-------|-------------|----------------|
| 10 | 0.9350 | 0.9619 |
| 20 | 0.9598 | 0.9624 |
| 30 | 0.9633 | 0.9607 |
| 40 | 0.9649 | 0.9618 |
| 50 | 0.9665 | 0.9623 |

## Final Performance
- **Train AUROC: 0.9665**
- **Held-out AUROC: 0.9623**

## Interpretation
The held-out AUROC of 0.9623 is evaluated on text from generators
never seen during training (Gemma-7B, Phi-3-mini, DeepSeek-7B). A held-out AUROC close to
or exceeding train AUROC supports the invariance claim — the probabilistic curvature features
generalize across generating models without being tied to any specific model's distribution.
