"""
Evaluation pipeline using RAGAS framework.
Compares baseline vs fine-tuned embedding models across retrieval and generation metrics.
"""

import json
import argparse
import yaml
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from retriever import FAISSRetriever
from generate import build_rag_chain


def load_eval_set(path: str) -> list[dict]:
    """Load evaluation queries with ground truth."""
    with open(path, "r") as f:
        return json.load(f)


def run_evaluation(
    eval_set: list[dict],
    use_finetuned: bool = False,
    config_path: str = "configs/config.yaml",
) -> dict:
    """Run RAG pipeline on eval set and compute RAGAS metrics."""
    chain, retriever = build_rag_chain(config_path, use_finetuned=use_finetuned)

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for item in eval_set:
        question = item["question"]
        gt = item["ground_truth"]

        # Retrieve
        results = retriever.retrieve(question)
        ctx = [r.chunk for r in results]

        # Generate
        answer = chain.invoke({"question": question})

        questions.append(question)
        answers.append(answer)
        contexts.append(ctx)
        ground_truths.append(gt)

    # Build RAGAS dataset
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    # Evaluate
    result = evaluate(
        eval_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    return result


def compare_models(eval_set: list[dict], config_path: str = "configs/config.yaml"):
    """Compare baseline vs fine-tuned embeddings."""
    print("=" * 60)
    print("Evaluating BASELINE model (all-MiniLM-L6-v2)")
    print("=" * 60)
    baseline_results = run_evaluation(eval_set, use_finetuned=False, config_path=config_path)
    print(baseline_results)

    print("\n" + "=" * 60)
    print("Evaluating FINE-TUNED model")
    print("=" * 60)
    finetuned_results = run_evaluation(eval_set, use_finetuned=True, config_path=config_path)
    print(finetuned_results)

    # Summary comparison
    comparison = pd.DataFrame({
        "Metric": ["Context Precision", "Faithfulness", "Answer Relevancy", "Context Recall"],
        "Baseline": [
            baseline_results["context_precision"],
            baseline_results["faithfulness"],
            baseline_results["answer_relevancy"],
            baseline_results["context_recall"],
        ],
        "Fine-tuned": [
            finetuned_results["context_precision"],
            finetuned_results["faithfulness"],
            finetuned_results["answer_relevancy"],
            finetuned_results["context_recall"],
        ],
    })
    comparison["Delta"] = comparison["Fine-tuned"] - comparison["Baseline"]

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(comparison.to_string(index=False))

    # Save results
    output = {
        "baseline": {k: float(v) for k, v in baseline_results.items()},
        "finetuned": {k: float(v) for k, v in finetuned_results.items()},
    }
    with open("evaluation/results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to evaluation/results.json")


def main():
    parser = argparse.ArgumentParser(description="RAGAS Evaluation Pipeline")
    parser.add_argument("--test_set", default="data/eval_queries.json")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--compare", action="store_true", help="Compare baseline vs fine-tuned")
    args = parser.parse_args()

    eval_set = load_eval_set(args.test_set)

    if args.compare:
        compare_models(eval_set, args.config)
    else:
        results = run_evaluation(eval_set, config_path=args.config)
        print(results)


if __name__ == "__main__":
    main()
