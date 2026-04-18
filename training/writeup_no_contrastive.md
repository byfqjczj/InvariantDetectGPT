# Detector Training Results (No Contrastive Loss)

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
| 10 | 0.8682 | 0.8675 |
| 20 | 0.9135 | 0.9475 |
| 30 | 0.9432 | 0.9595 |
| 40 | 0.9542 | 0.9617 |
| 50 | 0.9579 | 0.9602 |

## Final Performance
- **Train AUROC: 0.9579**
- **Held-out AUROC: 0.9602**
