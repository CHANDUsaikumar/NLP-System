"""LLM Model Evaluation & Scoring CLI Suite for Custom Datasets.

Evaluates candidate Hugging Face transformer models (GPT-2, GPT-Neo, T5, BART, DistilBERT, RoBERTa)
on custom user dataset files, computing ROUGE-1, ROUGE-2, ROUGE-L, latency, throughput, and memory scores.
"""

import sys
import json
import argparse
from pathlib import Path

# Ensure workspace root directory is on sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluation.model_evaluator import ModelEvaluator
from src.utils.logger import logger


def run_custom_dataset_evaluation(dataset_path: Path, task_filter: str = None):
    """Runs head-to-head comparative evaluation of LLM candidate models on a custom dataset file.

    Args:
        dataset_path (Path): Absolute or relative path to target JSON dataset file.
        task_filter (str, optional): Filter evaluation to a specific task key.

    Returns:
        dict: Serializable evaluation report containing scores and performance metrics.
    """
    print("\n" + "=" * 85)
    print(f" 📊 LLM MODEL EVALUATION ON CUSTOM DATA: {dataset_path.name}")
    print("=" * 85)

    evaluator = ModelEvaluator(dataset_path=dataset_path)
    report = evaluator.evaluate_all()

    print(
        f"{'Task Category':<25} | "
        f"{'Role':<8} | "
        f"{'Model Checkpoint':<40} | "
        f"{'Avg Latency':<12} | "
        f"{'Throughput':<12} | "
        f"{'ROUGE-L':<10} | "
        f"{'RAM (MB)':<10}"
    )
    print("-" * 135)

    serializable_report = {}

    for task_key, candidates in report.task_comparisons.items():
        if task_filter and task_key != task_filter:
            continue

        serializable_report[task_key] = {}

        for role, res in candidates.items():
            model_short = res.model_name
            if len(model_short) > 38:
                model_short = model_short[:35] + "..."

            print(
                f"{task_key:<25} | "
                f"{role.upper():<8} | "
                f"{model_short:<40} | "
                f"{res.avg_latency_ms:>8.1f} ms | "
                f"{res.avg_throughput_tps:>8.1f} t/s | "
                f"{res.avg_rougel:>8.4f}   | "
                f"{res.ram_usage_mb:>8.1f}"
            )

            serializable_report[task_key][role] = {
                "model_name": res.model_name,
                "role": res.role,
                "sample_count": res.sample_count,
                "avg_latency_ms": res.avg_latency_ms,
                "avg_throughput_tps": res.avg_throughput_tps,
                "ram_usage_mb": res.ram_usage_mb,
                "avg_rouge1": res.avg_rouge1,
                "avg_rouge2": res.avg_rouge2,
                "avg_rougel": res.avg_rougel,
                "outputs": res.sample_outputs
            }

    return serializable_report


def main():
    parser = argparse.ArgumentParser(description="LLM Model Evaluation & Scoring CLI for Custom Data")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(ROOT_DIR / "assets" / "evaluation_dataset.json"),
        help="Path to custom dataset JSON file (containing prompts and reference texts)."
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Optional task category filter (e.g. summarization, question_answering, text_generation, sentiment, named_entity_recognition, translation)."
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save model evaluation scores to logs/evaluation_report.json"
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = ROOT_DIR / dataset_path

    if not dataset_path.exists():
        print(f"❌ Error: Dataset file not found at {dataset_path}")
        sys.exit(1)

    scores_report = run_custom_dataset_evaluation(dataset_path=dataset_path, task_filter=args.task)

    if args.save_report:
        logs_dir = ROOT_DIR / "logs"
        logs_dir.mkdir(exist_ok=True)
        report_file = logs_dir / "evaluation_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(scores_report, f, indent=2)
        print(f"\n✅ Model evaluation report saved to {report_file}")

    print("\nLLM Model Evaluation completed successfully.\n")


if __name__ == "__main__":
    main()
