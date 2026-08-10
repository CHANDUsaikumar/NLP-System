"""Declarative Routing Rules and Keyword Definitions for Supported NLP Tasks."""

from typing import Dict, Any

ROUTING_RULES: Dict[str, Dict[str, Any]] = {
    "translation": {
        "task_name": "Translation",
        "keywords": [
            "translate", "translation", "translate to", "translate into",
            "in french", "in german", "in romanian"
        ],
        "pattern": r"(?i)\b(translate|translation|convert to|in french|in german|in romanian)\b",
        "primary_model": "t5-base",
        "fallback_model": "t5-small",
        "quality_metric": "BLEU"
    },
    "sentiment": {
        "task_name": "Sentiment Analysis",
        "keywords": [
            "sentiment", "sentiment analysis", "positive", "negative", "neutral", "emotion", "review",
            "amazing", "terrible", "horrible", "love", "hate", "awesome", "awful", "bad", "good"
        ],
        "pattern": r"(?i)\b(sentiment|sentiment analysis|positive|negative|neutral|emotion|review:|amazing|terrible|horrible|love|hate|awesome|awful|bad|good)\b",
        "primary_model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "fallback_model": "distilbert-base-uncased-finetuned-sst-2-english",
        "quality_metric": "Accuracy"
    },
    "summarization": {
        "task_name": "Summarization",
        "keywords": [
            "summarize", "summary", "summarize this", "shorten this", "give me a summary", "key points"
        ],
        "pattern": r"(?i)\b(summarize|summary|summarize this|shorten this|give me a summary|key points|tldr|synopsis)\b",
        "primary_model": "sshleifer/distilbart-cnn-12-6",
        "fallback_model": "t5-small",
        "quality_metric": "ROUGE-L"
    }
}
