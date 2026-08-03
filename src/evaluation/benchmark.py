"""Standardized benchmarking suite testing model routing latency and token throughput."""

from typing import List, Dict, Any
from src.router.dynamic_router import DynamicRouter
from src.utils.validators import UserRequestPayload
from src.utils.logger import logger


BENCHMARK_PROMPTS = [
    {
        "prompt": "Artificial intelligence has transformed modern software engineering by introducing dynamic model routing, automated code generation, and intelligent testing workflows. Companies deploy transformer models to optimize compute cost.",
        "expected_task": "summarization"
    },
    {
        "prompt": "I absolutely love using this new software! The user experience is incredibly smooth and responsive.",
        "expected_task": "sentiment"
    },
    {
        "prompt": "What is the capital of France and what is its primary historical significance?",
        "expected_task": "question_answering"
    },
    {
        "prompt": "Once upon a time in a distant galaxy powered by neural networks, a lonely satellite started broadcasting mysterious quantum signals.",
        "expected_task": "text_generation"
    }
]


class SystemBenchmark:
    """Runs a batch evaluation suite over standard benchmark prompts."""

    def __init__(self, router: DynamicRouter = None):
        self.router = router or DynamicRouter()

    def run_suite(self, prompts: List[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        test_prompts = prompts or BENCHMARK_PROMPTS
        results = []
        logger.info(f"Running System Benchmark Suite on {len(test_prompts)} prompts...")

        for idx, item in enumerate(test_prompts, 1):
            prompt = item["prompt"]
            expected = item.get("expected_task", "unknown")
            
            payload = UserRequestPayload(prompt=prompt)
            res = self.router.process_request(payload)
            
            is_match = (res.task == expected)
            results.append({
                "id": idx,
                "prompt_preview": prompt[:60] + "...",
                "expected_task": expected,
                "actual_task": res.task,
                "task_matched": is_match,
                "model_used": res.selected_model,
                "confidence": res.confidence_score,
                "latency_ms": res.latency_ms,
                "tokens_per_sec": res.token_throughput,
                "device": res.device_used
            })
            
        logger.info("System Benchmark Suite completed successfully.")
        return results
