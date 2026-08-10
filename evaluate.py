"""Simple CLI Evaluation Benchmark for Dynamic NLP Model Router.

Executes quantitative benchmarks comparing primary and fallback models across 3 core tasks:
- Summarization (Quality Metric: ROUGE-L)
- Sentiment Analysis (Quality Metric: Accuracy)
- Translation (Quality Metric: BLEU)
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


def run_benchmark(dataset_path: Path):
    """Executes model benchmarks on dataset and prints comparative results."""
    print("\n" + "=" * 95)
    print(" 📊 DYNAMIC NLP MODEL ROUTER BENCHMARK (PRIMARY vs FALLBACK)")
    print("=" * 95)

    evaluator = ModelEvaluator(dataset_path=dataset_path)
    benchmark_results = evaluator.evaluate_all()

    print(
        f"{'Task':<22} | "
        f"{'Status':<10} | "
        f"{'Model Checkpoint':<45} | "
        f"{'Metric Name':<12} | "
        f"{'Quality Score':<14} | "
        f"{'Latency (ms)':<12}"
    )
    print("-" * 125)

    serializable = []

    for res in benchmark_results:
        print(
            f"{res.task:<22} | "
            f"{res.status:<10} | "
            f"{res.model:<45} | "
            f"{res.quality_metric_name:<12} | "
            f"{res.quality_score:>13.4f}  | "
            f"{res.latency_ms:>10.1f} ms"
        )
        serializable.append({
            "task": res.task,
            "status": res.status,
            "model": res.model,
            "quality_metric_name": res.quality_metric_name,
            "quality_score": res.quality_score,
            "latency_ms": res.latency_ms
        })

    print("=" * 95 + "\n")
    return serializable


def main():
    parser = argparse.ArgumentParser(description="Dynamic NLP Model Router Benchmark CLI")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(ROOT_DIR / "assets" / "evaluation_dataset.json"),
        help="Path to evaluation dataset JSON file."
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save benchmark results to logs/benchmark_results.json"
    )

    args = parser.parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = ROOT_DIR / dataset_path

    if not dataset_path.exists():
        print(f"❌ Error: Dataset file not found at {dataset_path}")
        sys.exit(1)

    results = run_benchmark(dataset_path=dataset_path)

    if args.save_report:
        logs_dir = ROOT_DIR / "logs"
        logs_dir.mkdir(exist_ok=True)
        report_file = logs_dir / "benchmark_results.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"✅ Benchmark results saved to {report_file}\n")


if __name__ == "__main__":
    main()
