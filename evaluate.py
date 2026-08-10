"""Model and Router Evaluation Suite CLI for Adaptive NLP System.

Executes quantitative evaluations of the hybrid router (accuracy, precision, recall, F1,
confusion matrix) and runs system performance benchmarks (latency, throughput, memory).
"""

import sys
import json
import argparse
from pathlib import Path

# Ensure workspace root directory is on sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluation.router_evaluator import RouterEvaluator
from src.evaluation.benchmark import SystemBenchmark
from src.utils.logger import logger


def run_router_evaluation():
    """Runs router accuracy evaluation on labeled benchmark dataset."""
    print("\n" + "=" * 70)
    print(" 🎯 HYBRID INTENT ROUTER EVALUATION")
    print("=" * 70)

    evaluator = RouterEvaluator()
    report = evaluator.evaluate()

    print(f"Total Test Samples    : {report.total_samples}")
    print(f"Correct Predictions   : {report.correct_predictions}")
    print(f"Routing Accuracy      : {report.accuracy * 100:.2f}%")
    print(f"Average Confidence    : {report.average_confidence * 100:.2f}%")
    print(f"Average Router Latency: {report.average_latency_ms:.2f} ms")

    print("\n--- 📊 Macro & Weighted Classification Metrics ---")
    print(f"Macro Precision       : {report.precision_macro:.4f}")
    print(f"Macro Recall          : {report.recall_macro:.4f}")
    print(f"Macro F1-Score        : {report.f1_macro:.4f}")
    print(f"Weighted Precision    : {report.precision_weighted:.4f}")
    print(f"Weighted Recall       : {report.recall_weighted:.4f}")
    print(f"Weighted F1-Score     : {report.f1_weighted:.4f}")

    print("\n--- 🔀 Strategy Distribution ---")
    for strategy, count in report.strategy_distribution.items():
        pct = (count / report.total_samples) * 100
        print(f"  • {strategy:<28}: {count:2d} ({pct:.1f}%)")

    print("\n--- 📈 Per-Class Classification Breakdown ---")
    print(f"{'Task Category':<28} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 75)
    for task, metrics in report.per_class_metrics.items():
        print(
            f"{task:<28} | "
            f"{metrics['precision']:<10.4f} | "
            f"{metrics['recall']:<10.4f} | "
            f"{metrics['f1_score']:<10.4f} | "
            f"{metrics['support']:<8d}"
        )

    print("\n--- 🧩 Confusion Matrix ---")
    tasks = sorted(list(report.confusion_matrix.keys()))
    header = f"{'True \\ Pred':<25} | " + " | ".join(f"{t[:8]:<8}" for t in tasks)
    print(header)
    print("-" * len(header))
    for t_true in tasks:
        row = f"{t_true:<25} | "
        row += " | ".join(f"{report.confusion_matrix[t_true].get(t_pred, 0):<8d}" for t_pred in tasks)
        print(row)

    return report


def run_system_benchmark():
    """Runs system performance benchmark suite over candidate models."""
    print("\n" + "=" * 70)
    print(" 🚀 SYSTEM BENCHMARK SUITE")
    print("=" * 70)

    benchmark = SystemBenchmark()
    results = benchmark.run_suite()

    print(f"{'Prompt Preview':<35} | {'Detected Task':<20} | {'Selected Model':<25} | {'Latency':<10} | {'Throughput':<12}")
    print("-" * 110)

    for item in results:
        preview = item["prompt_preview"]
        if len(preview) > 33:
            preview = preview[:30] + "..."
        task = item["actual_task"]
        model = item["model_used"].split("/")[-1]
        latency = f"{item['latency_ms']:.1f} ms"
        throughput = f"{item['tokens_per_sec']:.1f} t/s"

        print(f"{preview:<35} | {task:<20} | {model:<25} | {latency:<10} | {throughput:<12}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Adaptive NLP System Model & Router Evaluation CLI")
    parser.add_argument(
        "--mode",
        choices=["router", "benchmark", "all"],
        default="all",
        help="Evaluation mode to execute: 'router' (intent classification accuracy), 'benchmark' (model performance), or 'all'."
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save evaluation report to logs/evaluation_report.json"
    )

    args = parser.parse_args()

    report_data = {}

    if args.mode in ["router", "all"]:
        router_report = run_router_evaluation()
        report_data["router_evaluation"] = {
            "total_samples": router_report.total_samples,
            "correct_predictions": router_report.correct_predictions,
            "accuracy": router_report.accuracy,
            "f1_macro": router_report.f1_macro,
            "f1_weighted": router_report.f1_weighted,
            "per_class_metrics": router_report.per_class_metrics,
            "strategy_distribution": router_report.strategy_distribution
        }

    if args.mode in ["benchmark", "all"]:
        benchmark_results = run_system_benchmark()
        report_data["system_benchmark"] = benchmark_results

    if args.save_report:
        logs_dir = ROOT_DIR / "logs"
        logs_dir.mkdir(exist_ok=True)
        report_file = logs_dir / "evaluation_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        print(f"\n✅ Evaluation report saved to {report_file}")

    print("\nEvaluation completed successfully.\n")


if __name__ == "__main__":
    main()
