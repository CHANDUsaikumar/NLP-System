"""FastAPI Web Server for Dynamic NLP Model Router."""

import sys
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

# Ensure workspace root directory is on sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.router.router import DynamicRouter
from src.evaluation.model_evaluator import ModelEvaluator
from src.utils.validators import UserRequestPayload, RouterResponsePayload
from src.utils.logger import logger
from config.settings import settings

UI_DIR = ROOT_DIR / "ui"

# Instantiate exportable FastAPI application for serverless / ASGI deployment (Vercel compatible)
app = FastAPI(
    title="Dynamic NLP Model Router API",
    description="Rule-based intent classification and benchmark-driven transformer model selection API.",
    version="1.0.0"
)

# Enable CORS for local development and cross-origin clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared Router and Evaluator singletons
router = DynamicRouter()
evaluator = ModelEvaluator()


@app.get("/health", response_model=Dict[str, str])
async def health_check() -> Dict[str, str]:
    """Health check endpoint returning system status."""
    return {"status": "ok"}


@app.post("/api/chat", response_model=RouterResponsePayload)
@app.post("/api/route", response_model=RouterResponsePayload)
async def process_chat(payload: UserRequestPayload) -> RouterResponsePayload:
    """Processes user prompt through intent classifier and executes selected model."""
    try:
        res = router.process(payload.prompt)
        return RouterResponsePayload(
            prompt=res.prompt,
            intent_detected=res.intent_detected,
            detected_task=res.detected_task,
            selected_model=res.selected_model,
            model_type=res.model_type,
            fallback_reason=res.fallback_reason,
            latency_ms=res.latency_ms,
            response_text=res.response_text,
            response=res.response_text
        )
    except Exception as e:
        logger.error(f"Error executing FastAPI chat route: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/benchmark", response_model=List[Dict[str, Any]])
async def get_benchmark() -> List[Dict[str, Any]]:
    """Returns primary vs fallback model evaluation metrics on benchmark test dataset."""
    try:
        results = evaluator.evaluate_all()
        return [
            {
                "task": r.task,
                "status": r.status,
                "model": r.model,
                "quality_metric_name": r.quality_metric_name,
                "quality_score": r.quality_score,
                "latency_ms": r.latency_ms
            }
            for r in results
        ]
    except Exception as e:
        logger.error(f"Error executing FastAPI benchmark route: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/evaluation-methodology", response_model=Dict[str, Any])
async def get_evaluation_methodology() -> Dict[str, Any]:
    """Returns evaluation methodology metadata, visual pipeline steps, models, and dataset sample counts."""
    try:
        metadata = settings.load_evaluation_metadata()
        registry = settings.load_model_registry().get("models", {})
        dataset = evaluator.load_dataset()
        
        sample_counts = {}
        for item in dataset:
            t = item.get("target_task")
            if t:
                sample_counts[t] = sample_counts.get(t, 0) + 1

        tasks_meta = metadata.get("tasks", {})
        for task_key, task_info in tasks_meta.items():
            reg_info = registry.get(task_key, {})
            task_info["primary_model"] = reg_info.get("model_name", "N/A")
            task_info["fallback_model"] = reg_info.get("fallback_model", "N/A")
            task_info["sample_count"] = sample_counts.get(task_key, 0)

        return {
            "visual_pipeline": metadata.get("visual_pipeline", []),
            "tasks": tasks_meta,
            "total_dataset_samples": len(dataset)
        }
    except Exception as e:
        logger.error(f"Error executing FastAPI evaluation-methodology route: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Mount Static UI Files at Root if UI directory exists
if UI_DIR.exists():
    app.mount("/", StaticFiles(directory=str(UI_DIR), html=True), name="ui")



import os
PORT = int(os.getenv("PORT", "7860"))

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=PORT, reload=False)
