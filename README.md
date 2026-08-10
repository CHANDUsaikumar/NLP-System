# LLM Model Evaluation & Scoring System for Custom Data 🤖📊

> **Resume Highlight**: Engineered an enterprise-grade LLM Evaluation & Scoring System leveraging PyTorch and Hugging Face Transformers. Designed a modular evaluation framework that benchmarks candidate LLMs strictly constrained to 6 Hugging Face architectures (`GPT-2`, `GPT-Neo`, `T5`, `BART`, `DistilBERT`, `RoBERTa`) on **custom user datasets**. Evaluates output quality scores (ROUGE-1, ROUGE-2, ROUGE-L, BLEU, Exact Match), inference latency, token throughput, and RAM memory footprint.

---

## 📌 Project Overview

The **LLM Model Evaluation & Scoring System** provides a standardized, scientific framework for evaluating and scoring different transformer LLMs on custom dataset files across key NLP tasks:

- **Summarization** (`sshleifer/distilbart-cnn-12-6` [BART], `t5-small` [T5])
- **Sentiment Analysis** (`cardiffnlp/twitter-roberta-base-sentiment-latest` [RoBERTa], `distilbert-base-uncased-finetuned-sst-2-english` [DistilBERT])
- **Question Answering** (`google/flan-t5-base` [T5], `google/flan-t5-small` [T5])
- **Text Generation** (`gpt2-medium` [GPT-2], `EleutherAI/gpt-neo-125M` [GPT-Neo])
- **Named Entity Recognition (NER)** (`elastic/distilbert-base-uncased-finetuned-conll03-english` [DistilBERT], `Jean-Baptiste/roberta-large-ner-english` [RoBERTa])
- **Translation** (`t5-base` [T5], `t5-small` [T5])

---

## 🏛️ System Architecture & Data Flow

```
                           ┌──────────────────────────┐
                           │   Custom JSON Dataset    │
                           │(Prompts & References)    │
                           └────────────┬─────────────┘
                                        │
                           ┌────────────▼─────────────┐
                           │   CLI Model Evaluator    │
                           │      (evaluate.py)       │
                           └────────────┬─────────────┘
                                        │
                           ┌────────────▼─────────────┐
                           │ Singleton Model Manager  │
                           │(Warm Memory LRU Cache)   │
                           └────────────┬─────────────┘
                                        │
        ┌──────────────────┬────────────┼────────────┬──────────────────┐
        │                  │            │            │                  │
┌───────▼────────┐ ┌───────▼──────┐ ┌───▼───────┐ ┌──▼────────────┐ ┌───▼───────────┐
│ Summarization  │ │  Sentiment   │ │    Q&A    │ │ Text Gen     │ │  NER / Trans  │
│ (DistilBART/T5)│ │(RoBERTa/D-Bert)│ │ (FLAN-T5) │ │(GPT-2/GPT-Neo)│ │(DistilBERT/T5)│
└───────┬────────┘ └───────┬──────┘ └───┬───────┘ └──┬────────────┘ └───┬───────────┘
        │                  │            │            │                  │
        └──────────────────┴────────────┼────────────┴──────────────────┘
                                        │
                           ┌────────────▼─────────────┐
                           │  Quantitative Scoring    │
                           │(ROUGE, BLEU, Latency, RAM)│
                           └────────────┬─────────────┘
                                        │
                           ┌────────────▼─────────────┐
                           │ JSON Evaluation Report   │
                           │(logs/evaluation_report)  │
                           └──────────────────────────┘
```

---

## ✨ Key Features

- **📊 Custom Dataset Evaluation**: Benchmark candidate LLMs on any user-provided dataset file (`--dataset custom_data.json`). Comes pre-loaded with an expanded **60-sample evaluation dataset** (`assets/evaluation_dataset.json`) with ground-truth reference texts.
- **📈 Quantitative Quality Scoring**: Computes ROUGE-1, ROUGE-2, ROUGE-L, BLEU, and Exact Match similarity scores comparing generated LLM outputs against ground-truth reference texts.
- **⚡ Performance & Memory Profiling**: Measures exact inference latency (ms), token throughput (tokens/sec), and process RSS memory footprint (RAM in MB) per model.
- **🛡️ 6 Allowed Hugging Face Model Families**: Constrained strictly to `GPT-2`, `GPT-Neo`, `T5`, `BART`, `DistilBERT`, and `RoBERTa` for clean, reliable evaluation.
- **📄 Comprehensive Evaluation Report Document**: Full empirical benchmarks, task-by-task trade-offs, and hardware performance breakdowns are documented in [docs/EVALUATION_REPORT.md](file:///Users/saikumarchandu/Desktop/NLP%20System/docs/EVALUATION_REPORT.md).

---

## 🛠️ Fair Head-to-Head Candidate Evaluation Scores Matrix

All candidate models for each task category are evaluated on the **exact same test dataset**, producing head-to-head performance scores:

| Task Category | Candidate Role | Model Checkpoint | HF Architecture | Avg Latency (ms) | Throughput (t/s) | ROUGE-1 | ROUGE-2 | ROUGE-L | RAM (MB) |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Summarization** | **Primary** | `sshleifer/distilbart-cnn-12-6` | BART | 1,423.5 ms | 58.0 t/s | 0.5120 | 0.2840 | **0.4410** | 1,326.4 MB |
| **Summarization** | **Fallback** | `t5-small` | T5 | 1,261.7 ms | 61.3 t/s | 0.4450 | 0.2180 | **0.3850** | 1,834.6 MB |
| **Sentiment Analysis** | **Primary** | `cardiffnlp/twitter-roberta-base-sentiment-latest` | RoBERTa | 36.9 ms | 1,125.5 t/s | N/A | N/A | **Classification** | 1,891.6 MB |
| **Sentiment Analysis** | **Fallback** | `distilbert-base-uncased-finetuned-sst-2-english` | DistilBERT | 16.6 ms | 1,794.2 t/s | N/A | N/A | **Classification** | 1,888.5 MB |
| **Question Answering** | **Primary** | `google/flan-t5-base` | T5 | 259.8 ms | 56.8 t/s | 0.6840 | 0.4520 | **0.6120** | 1,974.5 MB |
| **Question Answering** | **Fallback** | `google/flan-t5-small` | T5 | 1,279.6 ms | 64.5 t/s | 0.5910 | 0.3620 | **0.5210** | 3,682.7 MB |
| **Text Generation** | **Primary** | `gpt2-medium` | GPT-2 | 4,732.9 ms | 36.6 t/s | 0.3820 | 0.1940 | **0.3150** | 776.3 MB |
| **Text Generation** | **Fallback** | `EleutherAI/gpt-neo-125M` | GPT-Neo | 5,546.1 ms | 34.0 t/s | 0.3210 | 0.1510 | **0.2840** | 1,234.7 MB |
| **NER Extraction** | **Primary** | `elastic/distilbert-base-uncased-finetuned-conll03-english` | DistilBERT | 22.3 ms | 2,946.4 t/s | 0.8650 | 0.7420 | **0.8100** | 1,217.1 MB |
| **NER Extraction** | **Fallback** | `Jean-Baptiste/roberta-large-ner-english` | RoBERTa | 55.6 ms | 1,128.4 t/s | 0.8920 | 0.7810 | **0.8420** | 974.5 MB |
| **Translation** | **Primary** | `t5-base` | T5 | 782.6 ms | 34.9 t/s | 0.5420 | 0.3810 | **0.4850** | 1,185.2 MB |
| **Translation** | **Fallback** | `t5-small` | T5 | 776.8 ms | 45.3 t/s | 0.4720 | 0.3120 | **0.4120** | 1,502.3 MB |

---

## 📁 Repository Structure

```
nlp_system/
├── evaluate.py                 # Master CLI evaluation suite entrypoint
├── config/
│   ├── settings.py             # Pydantic BaseSettings config loader
│   └── model_registry.yaml     # Model candidates & task configuration
├── assets/
│   └── evaluation_dataset.json # Expanded 60-sample dataset (prompts & references)
├── docs/
│   └── EVALUATION_REPORT.md    # Comprehensive evaluation documentation report
├── src/
│   ├── models/                 # Specialized model pipelines & singleton manager
│   │   ├── base_pipeline.py    # Abstract base pipeline class
│   │   ├── summarizer.py       # Summarization pipeline (DistilBART / T5)
│   │   ├── sentiment.py        # Multi-class sentiment pipeline (RoBERTa / DistilBERT)
│   │   ├── text_gen.py         # Q&A (FLAN-T5) & Text Generation (GPT-2 / GPT-Neo)
│   │   ├── extra_pipelines.py  # NER (DistilBERT / RoBERTa) & Translation (T5) pipelines
│   │   └── model_manager.py    # Singleton warm LRU cache & device resolver
│   ├── evaluation/             # Metrics & evaluation suite
│   │   ├── metrics.py          # ROUGE, BLEU, BERTScore, Exact Match
│   │   ├── model_evaluator.py  # Head-to-head model evaluator for custom data
│   │   └── benchmark.py        # System latency & throughput suite
│   └── utils/                  # Telemetry, exception & payload schemas
│       ├── logger.py           # Clean console logger
│       ├── exceptions.py       # Domain exception taxonomy
│       └── validators.py       # Pydantic payload schemas
├── logs/                       # Generated evaluation JSON reports
│   └── evaluation_report.json
├── tests/                      # Automated Pytest suite
│   └── unit/
│       ├── test_pipelines.py
│       ├── test_metrics.py
│       └── test_model_evaluator.py
├── requirements.txt            # Production dependencies
├── pytest.ini                  # Pytest settings
└── README.md                   # System documentation
```

---

## ⚡ Setup & Quick Start

### 1. Prerequisites
- **Python 3.11+** installed. Verify with:
  ```bash
  python --version
  ```

### 2. Virtual Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Usage Guide

### 1. Evaluate Models on Default Custom Dataset (60 Samples)
```bash
python evaluate.py --save-report
```

### 2. Evaluate Models on a User-Provided Custom Dataset JSON
```bash
python evaluate.py --dataset path/to/my_custom_data.json --save-report
```

### 3. Filter Evaluation by Specific Task Category
```bash
python evaluate.py --task summarization
```

### 4. Execute Pytest Test Suite
```bash
pytest tests/unit/ -v
```

---

## 📄 Documentation

For in-depth empirical performance breakdowns, task-by-task accuracy trade-offs, and hardware memory consumption analysis, see the [EVALUATION_REPORT.md](file:///Users/saikumarchandu/Desktop/NLP%20System/docs/EVALUATION_REPORT.md).
