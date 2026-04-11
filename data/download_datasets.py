"""
Download and preprocess the three benchmark datasets.

For each dataset we extract human-written text passages that will serve as:
  - The negative (human) class for training
  - The prompt source for generating AI counterparts
"""

from datasets import load_dataset
import json
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_SAMPLES = 5000  # per dataset; trim to keep things manageable
MIN_WORDS = 50
MAX_WORDS = 300


def truncate(text: str, max_words: int = MAX_WORDS) -> str:
    words = text.split()
    return " ".join(words[:max_words])


def extract_xsum(max_samples: int = MAX_SAMPLES) -> list[dict]:
    """XSum: use the document (news article) as human text."""
    ds = load_dataset("EdinburghNLP/xsum", split="train")
    samples = []
    for row in ds:
        text = truncate(row["document"])
        if len(text.split()) >= MIN_WORDS:
            samples.append({
                "text": text,
                "domain": "news",
                "prompt": f"Write a news article about the following topic: {row['summary']}",
            })
        if len(samples) >= max_samples:
            break
    return samples


def extract_writingprompts(max_samples: int = MAX_SAMPLES) -> list[dict]:
    """WritingPrompts: use the story as human text, prompt as the generation prompt."""
    ds = load_dataset("euclaise/writingprompts", split="train")
    samples = []
    for row in ds:
        text = truncate(row["story"])
        if len(text.split()) >= MIN_WORDS:
            samples.append({
                "text": text,
                "domain": "creative",
                "prompt": row["prompt"],
            })
        if len(samples) >= max_samples:
            break
    return samples


def extract_squad(max_samples: int = MAX_SAMPLES) -> list[dict]:
    """SQuAD: use the Wikipedia context paragraph as human text."""
    ds = load_dataset("rajpurkar/squad", split="train")
    seen = set()
    samples = []
    for row in ds:
        text = truncate(row["context"])
        if text in seen or len(text.split()) < MIN_WORDS:
            continue
        seen.add(text)
        samples.append({
            "text": text,
            "domain": "wikipedia",
            "prompt": f"Write a Wikipedia-style paragraph about: {row['title']}",
        })
        if len(samples) >= max_samples:
            break
    return samples


def main():
    datasets = {
        "xsum": extract_xsum,
        "writingprompts": extract_writingprompts,
        "squad": extract_squad,
    }

    for name, fn in datasets.items():
        out_path = os.path.join(OUTPUT_DIR, f"{name}.json")
        if os.path.exists(out_path):
            print(f"{name}: already exists, skipping")
            continue
        print(f"Downloading {name}...")
        samples = fn()
        with open(out_path, "w") as f:
            json.dump(samples, f, indent=2)
        print(f"{name}: saved {len(samples)} samples -> {out_path}")


if __name__ == "__main__":
    main()
