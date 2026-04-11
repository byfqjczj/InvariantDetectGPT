import json
import matplotlib.pyplot as plt
import numpy as np

REF_MODELS = ["mistral-7b", "phi-3-mini", "qwen-7b"]
COLORS = ["steelblue", "tomato", "seagreen", "mediumpurple", "orange", "gray"]

# Load per-model feature files
per_model_data = {}
for m in REF_MODELS:
    with open(f"extracted/{m}.json") as f:
        per_model_data[m] = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Curvature by Source (Human vs each AI generator) under each Reference Model", fontsize=13)

for col, ref_model in enumerate(REF_MODELS):
    rows = per_model_data[ref_model]

    # Group by source model
    by_source: dict[str, list] = {}
    for r in rows:
        src = r["source_model"]
        by_source.setdefault(src, []).append(r["curvature"])

    # Put human first, then sort AI generators
    sources = ["human"] + sorted(k for k in by_source if k != "human")
    groups = [by_source[s] for s in sources]

    ax = axes[col]
    means = [np.mean(g) for g in groups]
    ax.bar(sources, means, color=COLORS[:len(sources)], alpha=0.7, edgecolor="black", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"Reference: {ref_model}")
    ax.set_ylabel("mean curvature")
    ax.tick_params(axis="x", rotation=25)
    # Scale y-axis to the means only, with a small margin
    margin = (max(means) - min(means)) * 0.3 or 0.05
    ax.set_ylim(min(means) - margin, max(means) + margin)

    print(f"\n{ref_model}:")
    for src, vals in zip(sources, groups):
        print(f"  {src:15s} mean: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

plt.tight_layout()
plt.savefig("extracted/features_visualization.png", dpi=150)
print("\nSaved to extracted/features_visualization.png")
