"""Standardized benchmarking suite testing model inference latency and token throughput."""

from typing import List, Dict, Any
from src.models.model_manager import ModelManager
from src.utils.logger import logger

BENCHMARK_PROMPTS = [
    {
        "prompt": "Artificial intelligence has transformed modern software engineering by introducing dynamic model routing, automated code generation, and intelligent testing workflows. Companies deploy transformer models to optimize compute cost.",
        "task_key": "summarization"
    },
    {
        "prompt": "I absolutely love using this new software! The user experience is incredibly smooth and responsive.",
        "task_key": "sentiment"
    },
    {
        "prompt": "What is the capital of France and what is its primary historical significance?",
        "task_key": "question_answering"
    },
    {
        "prompt": "Once upon a time in a distant galaxy powered by neural networks, a lonely satellite started broadcasting mysterious quantum signals.",
        "task_key": "text_generation"
    }
]


class SystemBenchmark:
    """Runs a batch evaluation suite over standard benchmark prompts using ModelManager."""

    def __init__(self, model_manager: ModelManager = None):
        self.model_manager = model_manager or ModelManager()

    def run_suite(self, prompts: List[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        test_prompts = prompts or BENCHMARK_PROMPTS
        results = []
        logger.info(f"Running System Benchmark Suite on {len(test_prompts)} prompts...")

        for idx, item in enumerate(test_prompts, 1):
            prompt = item["prompt"]
            task_key = item.get("task_key", "text_generation")
            
            try:
                pipeline = self.model_manager.get_pipeline(task_key)
                output_text, latency_ms, throughput = pipeline.run(prompt)
                results.append({
                    "id": idx,
                    "prompt_preview": prompt[:60] + "...",
                    "task_key": task_key,
                    "model_used": pipeline.model_name,
                    "latency_ms": latency_ms,
                    "tokens_per_sec": throughput,
                    "device": pipeline.device
                })
            except Exception as err:
                logger.error(f"Benchmark error for prompt #{idx}: {err}")
            
        logger.info("System Benchmark Suite completed successfully.")
        return results
