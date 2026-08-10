"""Model and Router Evaluation Suite CLI for Adaptive NLP System.

Executes quantitative evaluations of candidate models (fair head-to-head evaluation on identical test datasets),
hybrid router classification accuracy, precision, recall, F1, and system benchmarks.
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
from src.evaluation.model_evaluator import ModelEvaluator
from src.utils.logger import logger


def run_fair_model_evaluation():
    """Runs head-to-head comparative evaluation of primary and fallback models on identical task datasets."""
    print("\n" + "=" * 80)
    print(" ⚖️ FAIR CANDIDATE MODEL EVALUATION (IDENTICAL TEST DATASETS PER TASK)")
    print("=" * 80)

    evaluator = ModelEvaluator()
    report = evaluator.evaluate_all()

    print(f"{'Task':<25} | {'Candidate Role':<10} | {'Model Checkpoint':<45} | {'Avg Latency':<12} | {'Throughput':<12} | {'ROUGE-L':<10} | {'RAM (MB)':<10}")
    print("-" * 135)

    serializable_report = {}

    for task_key, candidates in report.task_comparisons.items():
        serializable_report[task_key] = {}
        for role, res in candidates.items():
            model_short = res.model_name
            if len(model_short) > 43:
                model_short = model_short[:40] + "..."

            print(
                f"{task_key:<25} | "
                f"{role.upper():<10} | "
                f"{model_short:<45} | "
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
                "avg_rougel": res.avg_rougel
            }

    return serializable_report


def run_router_evaluation():
    """Runs router accuracy evaluation on labeled benchmark dataset."""
    print("\n" + "=" * 80)
    print(" 🎯 HYBRID INTENT ROUTER EVALUATION")
    print("=" * 80)

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
    print("\n" + "=" * 80)
    print(" 🚀 SYSTEM BENCHMARK SUITE")
    print("=" * 80)

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
        choices=["models", "router", "benchmark", "all"],
        default="all",
        help="Evaluation mode to execute: 'models' (fair candidate model comparison), 'router' (intent classification accuracy), 'benchmark' (model performance), or 'all'."
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save evaluation report to logs/evaluation_report.json"
    )

    args = parser.parse_args()

    report_data = {}

    if args.mode in ["models", "all"]:
        models_report = run_fair_model_evaluation()
        report_data["fair_model_evaluation"] = models_report

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
