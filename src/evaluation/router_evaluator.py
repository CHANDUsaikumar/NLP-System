"""Router Evaluator computing accuracy, precision, recall, F1, and confusion matrix against labeled datasets."""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from collections import Counter, defaultdict

from src.router.decision_engine import DecisionEngine, RoutingDecision
from src.utils.logger import logger

DATASET_PATH = Path(__file__).resolve().parent.parent.parent / "assets" / "evaluation_dataset.json"


@dataclass
class RouterEvaluationReport:
    """Structured report containing detailed router evaluation metrics."""
    total_samples: int
    correct_predictions: int
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    average_confidence: float
    average_latency_ms: float
    strategy_distribution: Dict[str, int]
    per_class_metrics: Dict[str, Dict[str, float]]
    confusion_matrix: Dict[str, Dict[str, int]]
    detailed_results: List[Dict[str, Any]]


class RouterEvaluator:
    """Evaluates router prediction performance against ground-truth labeled benchmarks."""

    def __init__(self, decision_engine: Optional[DecisionEngine] = None, dataset_path: Optional[Path] = None):
        self.decision_engine = decision_engine or DecisionEngine()
        self.dataset_path = dataset_path or DATASET_PATH

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Loads labeled evaluation benchmark dataset."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found at {self.dataset_path}")

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def evaluate(self, dataset: Optional[List[Dict[str, Any]]] = None) -> RouterEvaluationReport:
        """Executes full router evaluation and calculates classification metrics."""
        test_cases = dataset if dataset is not None else self.load_dataset()
        if not test_cases:
            raise ValueError("Evaluation dataset is empty.")

        y_true = []
        y_pred = []
        confidences = []
        latencies = []
        strategies = Counter()
        detailed_results = []

        all_tasks = sorted(list(set(case["target_task"] for case in test_cases)))

        for case in test_cases:
            prompt = case["prompt"]
            target_task = case["target_task"]

            decision: RoutingDecision = self.decision_engine.route(prompt)

            pred_task = decision.task_key
            is_correct = (pred_task == target_task)

            y_true.append(target_task)
            y_pred.append(pred_task)
            confidences.append(decision.confidence)
            latencies.append(decision.routing_latency_ms)
            strategies[decision.strategy.value] += 1

            detailed_results.append({
                "id": case.get("id", ""),
                "prompt": prompt,
                "target_task": target_task,
                "predicted_task": pred_task,
                "is_correct": is_correct,
                "confidence": round(decision.confidence, 4),
                "strategy": decision.strategy.value,
                "rationale": decision.rationale,
                "latency_ms": decision.routing_latency_ms
            })

        total = len(test_cases)
        correct = sum(1 for c in detailed_results if c["is_correct"])
        accuracy = correct / total if total > 0 else 0.0

        # Confusion Matrix initialization
        confusion_matrix = {t_true: {t_pred: 0 for t_pred in all_tasks} for t_true in all_tasks}
        for yt, yp in zip(y_true, y_pred):
            if yt in confusion_matrix and yp in confusion_matrix[yt]:
                confusion_matrix[yt][yp] += 1
            else:
                # Handle unexpected tasks dynamically
                if yt not in confusion_matrix:
                    confusion_matrix[yt] = defaultdict(int)
                confusion_matrix[yt][yp] += 1

        # Per-class & Macro/Weighted Precision, Recall, F1
        per_class_metrics = {}
        macro_prec, macro_rec, macro_f1 = 0.0, 0.0, 0.0
        weighted_prec, weighted_rec, weighted_f1 = 0.0, 0.0, 0.0

        unique_classes = set(y_true).union(set(y_pred))
        num_classes = len(unique_classes)

        for task in unique_classes:
            tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == task and yp == task)
            fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != task and yp == task)
            fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == task and yp != task)
            support = sum(1 for yt in y_true if yt == task)

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            per_class_metrics[task] = {
                "support": support,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4)
            }

            macro_prec += prec
            macro_rec += rec
            macro_f1 += f1

            weighted_prec += prec * (support / total)
            weighted_rec += rec * (support / total)
            weighted_f1 += f1 * (support / total)

        if num_classes > 0:
            macro_prec /= num_classes
            macro_rec /= num_classes
            macro_f1 /= num_classes

        avg_conf = sum(confidences) / total if total > 0 else 0.0
        avg_lat = sum(latencies) / total if total > 0 else 0.0

        return RouterEvaluationReport(
            total_samples=total,
            correct_predictions=correct,
            accuracy=round(accuracy, 4),
            precision_macro=round(macro_prec, 4),
            recall_macro=round(macro_rec, 4),
            f1_macro=round(macro_f1, 4),
            precision_weighted=round(weighted_prec, 4),
            recall_weighted=round(weighted_rec, 4),
            f1_weighted=round(weighted_f1, 4),
            average_confidence=round(avg_conf, 4),
            average_latency_ms=round(avg_lat, 2),
            strategy_distribution=dict(strategies),
            per_class_metrics=per_class_metrics,
            confusion_matrix=confusion_matrix,
            detailed_results=detailed_results
        )
