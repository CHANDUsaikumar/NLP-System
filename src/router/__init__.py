"""Router package exposing DecisionEngine, DynamicRouter, Preprocessor, and Heuristic tools."""

from src.router.preprocessing import TextPreprocessor, PreprocessedInput
from src.router.heuristic_router import HeuristicRouter, HeuristicMatch
from src.router.zero_shot_router import ZeroShotRouter, ZeroShotResult
from src.router.rationale_generator import RationaleGenerator
from src.router.decision_engine import DecisionEngine, RoutingDecision, RoutingStrategy
from src.router.dynamic_router import DynamicRouter

__all__ = [
    "TextPreprocessor",
    "PreprocessedInput",
    "HeuristicRouter",
    "HeuristicMatch",
    "ZeroShotRouter",
    "ZeroShotResult",
    "RationaleGenerator",
    "DecisionEngine",
    "RoutingDecision",
    "RoutingStrategy",
    "DynamicRouter"
]
