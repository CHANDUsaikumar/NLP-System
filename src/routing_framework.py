"""Rule-Based Model-Routing Framework for Dynamic LLM Task Selection."""

import re
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from config.settings import settings


@dataclass
class RoutingResult:
    """Dataclass holding dynamic model-routing decision metadata."""
    task_key: str
    task_name: str
    selected_model: str
    fallback_model: str
    architecture_family: str
    confidence_score: float
    matched_rule_id: str
    matched_rule_description: str
    rationale: str
    benchmark_metrics: Dict[str, Any]


class RuleBasedRouter:
    """Deterministic Rule-Based Router evaluating syntactic features, intent keywords, and prompt patterns."""

    def __init__(self):
        self.registry = settings.load_model_registry().get("models", {})
        self.rules = self._initialize_rules()

    def _initialize_rules(self) -> List[Dict[str, Any]]:
        """Defines prioritized regex rules, keyword triggers, and structural heuristics."""
        return [
            {
                "id": "RULE_01_TRANSLATION",
                "task_key": "translation",
                "task_name": "Translation",
                "priority": 1,
                "pattern": r"(?i)\b(translate|convert to|traduis|traducir|in french|in spanish|in german)\b",
                "description": "Explicit translation directive or cross-lingual keyword detected.",
                "primary_model": "t5-base",
                "fallback_model": "t5-small",
                "family": "T5",
                "metrics": {"rouge_l": 0.4850, "latency_ms": 782.6, "throughput_tps": 34.9, "ram_mb": 1185.2}
            },
            {
                "id": "RULE_02_NER_EXTRACTION",
                "task_key": "named_entity_recognition",
                "task_name": "Named Entity Recognition",
                "priority": 2,
                "pattern": r"(?i)\b(extract entities|find names|identify organizations|who is in|extract named|ner:)\b",
                "description": "Explicit token classification or named entity extraction pattern detected.",
                "primary_model": "elastic/distilbert-base-uncased-finetuned-conll03-english",
                "fallback_model": "Jean-Baptiste/roberta-large-ner-english",
                "family": "DistilBERT",
                "metrics": {"rouge_l": 0.8100, "latency_ms": 22.3, "throughput_tps": 2946.4, "ram_mb": 1217.1}
            },
            {
                "id": "RULE_03_SUMMARIZATION",
                "task_key": "summarization",
                "task_name": "Summarization",
                "priority": 3,
                "pattern": r"(?i)\b(summarize|tldr|synopsis|summary|abstract|brief summary)\b",
                "description": "Explicit summarization command or multi-paragraph long document detected.",
                "primary_model": "sshleifer/distilbart-cnn-12-6",
                "fallback_model": "t5-small",
                "family": "BART",
                "metrics": {"rouge_l": 0.4410, "latency_ms": 1423.5, "throughput_tps": 58.0, "ram_mb": 1326.4}
            },
            {
                "id": "RULE_04_SENTIMENT_ANALYSIS",
                "task_key": "sentiment",
                "task_name": "Sentiment Analysis",
                "priority": 4,
                "pattern": r"(?i)\b(sentiment:|review:|amazing|terrible|horrible|love|hate|awful|outstanding|subpar|frustrated)\b",
                "description": "Product/service review phrasing or strong emotional polarity words detected.",
                "primary_model": "distilbert-base-uncased-finetuned-sst-2-english",
                "fallback_model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "family": "DistilBERT",
                "metrics": {"rouge_l": "Classification", "latency_ms": 16.6, "throughput_tps": 1794.2, "ram_mb": 1888.5}
            },
            {
                "id": "RULE_05_QUESTION_ANSWERING",
                "task_key": "question_answering",
                "task_name": "Question Answering",
                "priority": 5,
                "pattern": r"(?i)^\s*(what|why|how|who|where|when|is|can|does|which)\b|\?$",
                "description": "Interrogative starter keyword or trailing question mark detected.",
                "primary_model": "google/flan-t5-base",
                "fallback_model": "google/flan-t5-small",
                "family": "T5",
                "metrics": {"rouge_l": 0.6120, "latency_ms": 259.8, "throughput_tps": 56.8, "ram_mb": 1974.5}
            },
            {
                "id": "RULE_06_TEXT_GENERATION",
                "task_key": "text_generation",
                "task_name": "Creative Text Generation",
                "priority": 6,
                "pattern": r"(?i)\b(write a story|generate a poem|compose|once upon a time|creative|draft|continue)\b",
                "description": "Open-ended creative generation directive or storytelling prompt detected.",
                "primary_model": "gpt2-medium",
                "fallback_model": "EleutherAI/gpt-neo-125M",
                "family": "GPT-2",
                "metrics": {"rouge_l": 0.3150, "latency_ms": 4732.9, "throughput_tps": 36.6, "ram_mb": 776.3}
            }
        ]

    def route(self, prompt: str) -> RoutingResult:
        """Evaluates input prompt against prioritized rule set and returns dynamic routing decision.

        Args:
            prompt (str): Input user prompt string.

        Returns:
            RoutingResult: Decision metadata including selected model, task, confidence, and rationale.
        """
        clean_prompt = prompt.strip()
        word_count = len(clean_prompt.split())

        # Check explicit rule patterns by priority
        for rule in self.rules:
            if re.search(rule["pattern"], clean_prompt):
                return RoutingResult(
                    task_key=rule["task_key"],
                    task_name=rule["task_name"],
                    selected_model=rule["primary_model"],
                    fallback_model=rule["fallback_model"],
                    architecture_family=rule["family"],
                    confidence_score=0.96,
                    matched_rule_id=rule["id"],
                    matched_rule_description=rule["description"],
                    rationale=f"Matched {rule['id']} ({rule['description']}). Dynamically routed prompt to specialized model '{rule['primary_model']}' ({rule['family']} Architecture). Benchmark Latency: {rule['metrics']['latency_ms']}ms, Throughput: {rule['metrics']['throughput_tps']} t/s.",
                    benchmark_metrics=rule["metrics"]
                )

        # Structural Heuristic Fallback: Long text (> 80 words) defaults to Summarization
        if word_count > 80:
            rule = self.rules[2]  # Summarization
            return RoutingResult(
                task_key="summarization",
                task_name="Summarization",
                selected_model=rule["primary_model"],
                fallback_model=rule["fallback_model"],
                architecture_family=rule["family"],
                confidence_score=0.88,
                matched_rule_id="HEURISTIC_LONG_TEXT",
                matched_rule_description=f"Long document heuristic ({word_count} words > 80 threshold).",
                rationale=f"Document size heuristic ({word_count} words) classified document for Summarization. Selected model '{rule['primary_model']}'. Benchmark Latency: {rule['metrics']['latency_ms']}ms.",
                benchmark_metrics=rule["metrics"]
            )

        # Default Fallback: Open Instruction Q&A (FLAN-T5)
        default_rule = self.rules[4]  # Question Answering
        return RoutingResult(
            task_key="question_answering",
            task_name="Question Answering / Instruction",
            selected_model=default_rule["primary_model"],
            fallback_model=default_rule["fallback_model"],
            architecture_family=default_rule["family"],
            confidence_score=0.75,
            matched_rule_id="DEFAULT_INSTRUCTION_FALLBACK",
            matched_rule_description="General instruction prompt defaulting to FLAN-T5 Base.",
            rationale=f"General instruction prompt routed to primary instruction model '{default_rule['primary_model']}' ({default_rule['family']}).",
            benchmark_metrics=default_rule["metrics"]
        )
