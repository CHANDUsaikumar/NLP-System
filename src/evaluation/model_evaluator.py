"""Model Evaluator comparing Primary and Fallback models on real evaluation datasets."""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from config.settings import settings
from src.models.model_manager import ModelManager
from src.evaluation.metrics import EvaluationMetrics
from src.utils.logger import logger

DATASET_PATH = Path(__file__).resolve().parent.parent.parent / "assets" / "evaluation_dataset.json"


@dataclass
class ModelBenchmarkResult:
    """Benchmark result for a single candidate model on a task."""
    task: str
    model: str
    quality_metric_name: str
    quality_score: float
    latency_ms: float
    status: str  # "Primary" or "Fallback"


class ModelEvaluator:
    """Evaluates primary and fallback models for Summarization, Sentiment Analysis, and Translation."""

    def __init__(self, dataset_path: Optional[Path] = None):
        self.dataset_path = dataset_path or DATASET_PATH
        self.model_manager = ModelManager()
        self.registry = settings.load_model_registry().get("models", {})

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Loads test dataset JSON."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found at {self.dataset_path}")
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def evaluate_model(
        self,
        task_key: str,
        model_name: str,
        status: str,
        test_samples: List[Dict[str, Any]]
    ) -> ModelBenchmarkResult:
        """Evaluates a single model candidate on test samples and calculates real benchmark values."""
        logger.info(f"Evaluating {status.upper()} model '{model_name}' for task '{task_key}' on {len(test_samples)} samples...")
        
        try:
            pipeline = self.model_manager.get_pipeline_by_name(task_key, model_name)
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            metric_name = "ROUGE-L" if task_key == "summarization" else ("Accuracy" if task_key == "sentiment" else "BLEU")
            return ModelBenchmarkResult(
                task=task_key,
                model=model_name,
                quality_metric_name=metric_name,
                quality_score=0.0,
                latency_ms=0.0,
                status=status
            )

        latencies = []
        quality_scores = []

        if task_key == "sentiment":
            y_true = [s.get("reference_text", "") for s in test_samples]
            y_pred = []
            for sample in test_samples:
                prompt = sample["prompt"]
                try:
                    out_text, lat = pipeline.run(prompt)
                    latencies.append(lat)
                    y_pred.append(out_text)
                except Exception as err:
                    logger.warning(f"Inference error on sample for '{model_name}': {err}")

            accuracy = EvaluationMetrics.compute_accuracy(y_true, y_pred)
            quality_scores.append(accuracy)
            metric_name = "Accuracy"

        elif task_key == "summarization":
            metric_name = "ROUGE-L"
            for sample in test_samples:
                prompt = sample["prompt"]
                ref = sample.get("reference_text", "")
                try:
                    out_text, lat = pipeline.run(prompt)
                    latencies.append(lat)
                    if ref:
                        score = EvaluationMetrics.compute_rouge_l(out_text, ref)
                        quality_scores.append(score)
                except Exception as err:
                    logger.warning(f"Inference error on sample for '{model_name}': {err}")

        elif task_key == "translation":
            metric_name = "BLEU"
            for sample in test_samples:
                prompt = sample["prompt"]
                ref = sample.get("reference_text", "")
                try:
                    out_text, lat = pipeline.run(prompt)
                    latencies.append(lat)
                    if ref:
                        score = EvaluationMetrics.compute_bleu(out_text, ref)
                        quality_scores.append(score)
                except Exception as err:
                    logger.warning(f"Inference error on sample for '{model_name}': {err}")
        else:
            metric_name = "Score"

        n = max(len(latencies), 1)
        avg_latency = sum(latencies) / n
        avg_quality = (sum(quality_scores) / max(len(quality_scores), 1)) if quality_scores else 0.0

        return ModelBenchmarkResult(
            task=task_key,
            model=model_name,
            quality_metric_name=metric_name,
            quality_score=round(avg_quality, 4),
            latency_ms=round(avg_latency, 2),
            status=status
        )

    def evaluate_all(self) -> List[ModelBenchmarkResult]:
        """Evaluates primary and fallback models for all 3 supported tasks."""
        dataset = self.load_dataset()
        task_samples = {}
        for item in dataset:
            t = item["target_task"]
            task_samples.setdefault(t, []).append(item)

        results = []

        for task_key in ["summarization", "sentiment", "translation"]:
            config = self.registry.get(task_key, {})
            samples = task_samples.get(task_key, [])
            if not samples:
                continue

            primary_model = config.get("model_name")
            fallback_model = config.get("fallback_model")

            if primary_model:
                results.append(self.evaluate_model(task_key, primary_model, "Primary", samples))

            if fallback_model and fallback_model != primary_model:
                results.append(self.evaluate_model(task_key, fallback_model, "Fallback", samples))

        return results
