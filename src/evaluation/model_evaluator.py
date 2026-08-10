"""Fair Model Evaluator evaluating candidate models head-to-head on identical task datasets."""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from config.settings import settings
from src.models.model_manager import ModelManager
from src.evaluation.metrics import EvaluationMetrics
from src.utils.logger import logger, get_memory_usage_mb

DATASET_PATH = Path(__file__).resolve().parent.parent.parent / "assets" / "evaluation_dataset.json"


@dataclass
class CandidateModelResult:
    """Evaluation result for a single candidate model on a task dataset."""
    task_key: str
    model_name: str
    role: str  # "primary" or "fallback"
    sample_count: int
    avg_latency_ms: float
    avg_throughput_tps: float
    ram_usage_mb: float
    avg_rouge1: float
    avg_rouge2: float
    avg_rougel: float
    sample_outputs: List[Dict[str, Any]]


@dataclass
class FairModelEvaluationReport:
    """Comparative report containing head-to-head metrics for candidate models across all tasks."""
    task_comparisons: Dict[str, Dict[str, CandidateModelResult]]


class ModelEvaluator:
    """Evaluates primary and fallback candidate models on identical test datasets per task."""

    def __init__(self, dataset_path: Optional[Path] = None):
        self.dataset_path = dataset_path or DATASET_PATH
        self.model_manager = ModelManager()
        self.registry = settings.load_model_registry().get("models", {})

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Loads test dataset with ground-truth reference texts."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found at {self.dataset_path}")

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def evaluate_candidate(
        self,
        task_key: str,
        model_name: str,
        role: str,
        test_samples: List[Dict[str, Any]],
        task_config: Dict[str, Any]
    ) -> CandidateModelResult:
        """Evaluates a single model candidate on a specified list of test samples.

        Args:
            task_key (str): Task category identifier.
            model_name (str): Hugging Face checkpoint identifier.
            role (str): Model role ('primary' or 'fallback').
            test_samples (List[Dict[str, Any]]): Test samples for this task.
            task_config (Dict[str, Any]): Task configuration dictionary.

        Returns:
            CandidateModelResult: Aggregated evaluation result.
        """
        logger.info(f"Evaluating {role.upper()} model candidate '{model_name}' for task '{task_key}' on {len(test_samples)} samples...")

        # Instantiate pipeline directly via ModelManager instantiation logic
        pipeline_cfg = dict(task_config)
        pipeline_cfg["model_name"] = model_name

        try:
            pipeline_instance = self.model_manager._instantiate_pipeline(task_key, pipeline_cfg)
            pipeline_instance.load_pipeline()
        except Exception as e:
            logger.error(f"Failed to load candidate model '{model_name}': {e}")
            return CandidateModelResult(
                task_key=task_key,
                model_name=model_name,
                role=role,
                sample_count=len(test_samples),
                avg_latency_ms=0.0,
                avg_throughput_tps=0.0,
                ram_usage_mb=get_memory_usage_mb(),
                avg_rouge1=0.0,
                avg_rouge2=0.0,
                avg_rougel=0.0,
                sample_outputs=[]
            )

        latencies = []
        throughputs = []
        rouge1_scores = []
        rouge2_scores = []
        rougel_scores = []
        sample_outputs = []

        for sample in test_samples:
            prompt = sample["prompt"]
            reference = sample.get("reference_text")

            try:
                output_text, latency_ms, throughput = pipeline_instance.run(prompt)
                latencies.append(latency_ms)
                throughputs.append(throughput)

                rouge1, rouge2, rougel = 0.0, 0.0, 0.0
                if reference:
                    metrics = EvaluationMetrics.compute_all(output_text, reference)
                    rouge1 = metrics.get("rouge1", 0.0)
                    rouge2 = metrics.get("rouge2", 0.0)
                    rougel = metrics.get("rougeL", 0.0)

                rouge1_scores.append(rouge1)
                rouge2_scores.append(rouge2)
                rougel_scores.append(rougel)

                sample_outputs.append({
                    "id": sample.get("id", ""),
                    "prompt": prompt[:50] + "...",
                    "output": output_text,
                    "latency_ms": latency_ms,
                    "throughput": throughput,
                    "rougeL": rougel
                })

            except Exception as err:
                logger.warning(f"Inference error on sample for model '{model_name}': {err}")

        n = max(len(latencies), 1)
        avg_lat = sum(latencies) / n
        avg_tps = sum(throughputs) / n
        avg_r1 = sum(rouge1_scores) / n
        avg_r2 = sum(rouge2_scores) / n
        avg_rl = sum(rougel_scores) / n
        mem_mb = get_memory_usage_mb()

        return CandidateModelResult(
            task_key=task_key,
            model_name=model_name,
            role=role,
            sample_count=len(test_samples),
            avg_latency_ms=round(avg_lat, 2),
            avg_throughput_tps=round(avg_tps, 2),
            ram_usage_mb=round(mem_mb, 2),
            avg_rouge1=round(avg_r1, 4),
            avg_rouge2=round(avg_r2, 4),
            avg_rougel=round(avg_rl, 4),
            sample_outputs=sample_outputs
        )

    def evaluate_all(self) -> FairModelEvaluationReport:
        """Runs fair head-to-head candidate evaluation across all tasks in model registry."""
        dataset = self.load_dataset()
        task_samples = {}
        for item in dataset:
            t = item["target_task"]
            task_samples.setdefault(t, []).append(item)

        task_comparisons = {}

        for task_key, config in self.registry.items():
            samples = task_samples.get(task_key, [])
            if not samples:
                logger.warning(f"No test samples found in dataset for task '{task_key}'")
                continue

            primary_model = config.get("model_name")
            fallback_model = config.get("fallback_model")

            comparisons = {}
            if primary_model:
                comparisons["primary"] = self.evaluate_candidate(
                    task_key=task_key,
                    model_name=primary_model,
                    role="primary",
                    test_samples=samples,
                    task_config=config
                )

            if fallback_model and fallback_model != primary_model:
                comparisons["fallback"] = self.evaluate_candidate(
                    task_key=task_key,
                    model_name=fallback_model,
                    role="fallback",
                    test_samples=samples,
                    task_config=config
                )

            task_comparisons[task_key] = comparisons

        return FairModelEvaluationReport(task_comparisons=task_comparisons)
