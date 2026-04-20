"""
Fine-tune a sentence transformer model on the NLP abstracts corpus.
Uses contrastive learning with in-batch negatives to improve domain-specific retrieval.
"""

import json
import random
import argparse
import yaml
from pathlib import Path
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    evaluation,
)
from torch.utils.data import DataLoader


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_training_pairs(corpus_path: str, num_pairs: int = 2000) -> list[InputExample]:
    """
    Build training pairs from the corpus.
    Positive pairs: chunks from the same paper.
    Negative pairs: chunks from different papers (via in-batch negatives).
    """
    with open(corpus_path, "r") as f:
        papers = json.load(f)

    # Group abstracts by paper — use title+abstract as anchor/positive
    examples = []
    for paper in papers:
        if not paper.get("abstract") or not paper.get("title"):
            continue

        # Pair: title → abstract (positive)
        examples.append(InputExample(
            texts=[paper["title"], paper["abstract"]],
            label=1.0,
        ))

        # Pair: first half of abstract → second half (positive)
        abstract = paper["abstract"]
        mid = len(abstract) // 2
        if mid > 50:
            examples.append(InputExample(
                texts=[abstract[:mid], abstract[mid:]],
                label=1.0,
            ))

    random.shuffle(examples)
    return examples[:num_pairs]


def fine_tune(config_path: str = "configs/config.yaml"):
    """Fine-tune the sentence transformer on domain data."""
    config = load_config(config_path)
    ft_config = config["fine_tuning"]

    model_name = config["embedding"]["base_model"]
    output_path = config["embedding"]["fine_tuned_model"]
    corpus_path = config["data"]["corpus_path"]

    print(f"Loading base model: {model_name}")
    model = SentenceTransformer(model_name)

    print("Building training pairs...")
    train_examples = build_training_pairs(corpus_path)
    print(f"Created {len(train_examples)} training pairs")

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=ft_config["batch_size"],
    )

    # Multiple Negatives Ranking Loss — effective for retrieval fine-tuning
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Train
    print(f"Fine-tuning for {ft_config['epochs']} epochs...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=ft_config["epochs"],
        warmup_steps=int(len(train_dataloader) * ft_config["warmup_ratio"]),
        show_progress_bar=True,
        output_path=output_path,
    )

    print(f"Fine-tuned model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Sentence Transformer")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    fine_tune(args.config)


if __name__ == "__main__":
    main()
