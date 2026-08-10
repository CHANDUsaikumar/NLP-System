# Dynamic NLP Model Router 🤖⚡

A clean, modular **Dynamic NLP Model Router** that classifies task intents and dynamically routes user prompts to specialized Hugging Face transformer models based on empirical benchmark evaluation and primary/fallback availability policies.

---

## 📌 Supported NLP Tasks & Model Registry

The system is streamlined to support **3 core NLP tasks**, comparing a **Primary** and **Fallback** model candidate for each task:

| Task | Primary Model | Fallback Model | Quality Metric | Efficiency Metric |
| :--- | :--- | :--- | :--- | :--- |
| **Summarization** | `sshleifer/distilbart-cnn-12-6` | `t5-small` | `ROUGE-L` | Inference Latency (ms) |
| **Sentiment Analysis** | `cardiffnlp/twitter-roberta-base-sentiment-latest` | `distilbert-base-uncased-finetuned-sst-2-english` | `Accuracy` | Inference Latency (ms) |
| **Translation** | `t5-base` | `t5-small` | `BLEU` | Inference Latency (ms) |

---

## 🏛️ System Architecture

```text
                           ┌──────────────────────────┐
                           │    User Input Prompt     │
                           └────────────┬─────────────┘
                                        │
                           ┌────────────▼─────────────┐
                           │    Intent Classifier     │
                           │(src/router/intent_class) │
                           └────────────┬─────────────┘
                                        │
             ┌──────────────────────────┼──────────────────────────┐
             │                          │                          │
             ▼                          ▼                          ▼
   ┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
   │  Summarization   │       │Sentiment Analysis│       │   Translation    │
   │ (DistilBART / T5)│       │ (RoBERTa / D-Bert│       │  (T5-base / T5)  │
   └─────────┬────────┘       └─────────┬────────┘       └─────────┬────────┘
             │                          │                          │
             └──────────────────────────┼──────────────────────────┘
                                        │
                           ┌────────────▼─────────────┐
                           │ Primary / Fallback Router│
                           │   (src/router/router)    │
                           └────────────┬─────────────┘
                                        │
                           ┌────────────▼─────────────┐
                           │   Web UI & Response      │
                           │   (ui/ & server.py)      │
                           └──────────────────────────┘
```

---

## 🚀 Quick Start & Usage

### 1. Launch FastAPI Web Server
```bash
python server.py
```
> Starts the FastAPI + Uvicorn server locally at **`http://localhost:8000`**. The ASGI application instance (`app`) is directly importable from `server.py` for serverless or Vercel deployments.

### 2. Verify API Health Endpoint
```bash
curl http://localhost:8000/health
# Output: {"status": "ok"}
```

### 2. Run CLI Benchmark Evaluation
```bash
python evaluate.py --save-report
```

### 3. Run Pytest Suite
```bash
pytest tests/unit/ -v
```

---

## ⚙️ Routing Rules & Intent Keywords

- **Translation**: `translate`, `translation`, `translate to`, `translate into`, `in hindi`, `in french`, `in spanish`, `in telugu`, `in german`
- **Sentiment Analysis**: `sentiment`, `sentiment analysis`, `positive`, `negative`, `neutral`, `emotion`, `review:`, `amazing`, `terrible`
- **Summarization**: `summarize`, `summary`, `summarize this`, `shorten this`, `give me a summary`, `key points`

If no intent is detected, the router cleanly returns:
> *"Could not detect task intent. Please specify whether you want Summarization, Sentiment Analysis, or Translation."*

---

## 📊 Benchmark Comparison Table

| Task | Primary Model | Primary Quality | Primary Latency | Fallback Model | Fallback Quality | Fallback Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Summarization** | `sshleifer/distilbart-cnn-12-6` | ROUGE-L: `0.4146` | 1,649.4 ms | `t5-small` | ROUGE-L: `0.4054` | 1,264.9 ms |
| **Sentiment Analysis** | `cardiffnlp/twitter-roberta-base-sentiment-latest` | Accuracy: `100.0%` | 245.9 ms | `distilbert-base-uncased-finetuned-sst-2-english` | Accuracy: `80.0%` | 18.5 ms |
| **Translation** | `t5-base` | BLEU: `0.6264` | 614.1 ms | `t5-small` | BLEU: `0.5638` | 489.1 ms |

---

## 📁 Project Structure

```
nlp_system/
├── config/
│   ├── config.py             # Central configuration loader
│   └── model_registry.yaml   # Registry for Summarization, Sentiment, and Translation
├── assets/
│   └── evaluation_dataset.json # Test cases for the 3 core tasks
├── src/
│   ├── router/
│   │   ├── intent_classifier.py  # Keyword-based intent classification
│   │   ├── routing_rules.py      # Declarative routing rule patterns
│   │   └── router.py             # Router with primary/fallback execution policy
│   ├── models/
│   │   ├── base_pipeline.py      # Abstract transformer pipeline class
│   │   ├── summarization.py      # Summarization pipeline (DistilBART / T5-small)
│   │   ├── sentiment.py          # Sentiment pipeline (RoBERTa / DistilBERT)
│   │   ├── translation.py        # Translation pipeline with target language extraction
│   │   └── model_manager.py      # Singleton LRU warm loader
│   └── evaluation/
│       ├── metrics.py            # ROUGE-L, Accuracy, BLEU metrics
│       └── model_evaluator.py    # Benchmark evaluator for the 3 tasks
├── ui/                           # Simplified Web UI Frontend
│   ├── index.html                # Chatbot, Model Benchmark, and Routing Rules tabs
│   ├── styles.css                # Clean dark-mode styles
│   └── app.js                    # Web UI logic
├── evaluate.py                   # Simple CLI benchmark script
├── server.py                     # Web server (http://localhost:8000)
└── README.md                     # Documentation
```
