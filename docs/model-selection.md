# Model Selection Documentation

## Overview

This document explains **why each model was selected** for the three core tasks in the NLP System: **Summarization**, **Sentiment Analysis**, and **Translation**.  The selection is based on a combination of:

- **Documented model capabilities** (architecture, pre‑training, fine‑tuning).
- **Empirical results** from the internal benchmark (quality metric & latency).
- **Engineering considerations** such as hardware footprint and fallback reliability.

> **Note:** All statements refer to the *evaluation performed by this repository*; they do not claim global superiority.

---

### Summarization

**Candidate Models**
- Primary: `sshleifer/distilbart-cnn-12-6`
- Fallback: `t5-small`

**Why These Models Were Considered**
- **DistilBART‑CNN** – a distilled version of BART fine‑tuned on the CNN/DailyMail summarization corpus.  It is lightweight, fast, and specifically engineered for abstractive summarization.
- **T5‑small** – a general‑purpose encoder‑decoder model that can be fine‑tuned for summarization; used as a safety‑net when the primary model fails or exceeds latency budgets.

**Evaluation Criteria**
- **Quality Metric:** ROUGE‑L (longest‑common‑subsequence F1).
- **Latency:** average inference time per prompt (ms).
- **Resource Footprint:** RAM usage reported in the benchmark.

**Comparative Results** (from `logs/benchmark_results.json`)
| Model | ROUGE‑L Score | Avg Latency (ms) | Role |
|---|---:|---:|---|
| `sshleifer/distilbart-cnn-12-6` | 0.4146 | 1326.5 | Primary |
| `t5-small` | 0.4054 | 1064.6 | Fallback |

**Primary Model Decision**
- The DistilBART candidate achieved a **higher ROUGE‑L score** (0.4146 > 0.4054) while staying within acceptable latency limits.  Its architecture is purpose‑built for summarization, making it the **best‑performing candidate in this evaluation**.

**Fallback Model Decision**
- `t5-small` provides a comparable latency and a modest drop in ROUGE‑L.  It is retained as a fallback to guarantee service continuity if the primary model is unavailable or exceeds resource caps.

**Trade‑offs**
- DistilBART offers better summarization quality but requires slightly more memory (≈ 1.3 GB) than `t5-small` (≈ 1.5 GB).  The fallback sacrifices a small amount of ROUGE‑L for a marginally lower memory footprint.

---

### Sentiment Analysis

**Candidate Models**
- Primary: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- Fallback: `distilbert-base-uncased-finetuned-sst-2-english`

**Why These Models Were Considered**
- **RoBERTa‑base (Twitter‑sentiment)** – pre‑trained on a large corpus of tweets and fine‑tuned for three‑way sentiment classification.  It reliably captures informal language nuances.
- **DistilBERT‑SST‑2** – a distilled BERT model fine‑tuned on the Stanford Sentiment Treebank, offering a lightweight alternative.

**Evaluation Criteria**
- **Quality Metric:** Accuracy (exact‑match against the reference label).
- **Latency:** average inference time per prompt.
- **Robustness:** ability to handle neutral classes present in the dataset.

**Comparative Results**
| Model | Accuracy | Avg Latency (ms) | Role |
|---|---:|---:|---|
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | 1.00 | 39.84 | Primary |
| `distilbert-base-uncased-finetuned-sst-2-english` | 0.80 | 13.42 | Fallback |

**Primary Model Decision**
- The RoBERTa‑based model achieved **perfect accuracy** (1.00) on the held‑out set, outperforming the fallback.  Its performance on informal text aligns with the dataset characteristics, making it the **best‑performing candidate in this evaluation**.

**Fallback Model Decision**
- `distilbert-base-uncased-finetuned-sst-2-english` is considerably faster (≈ 13 ms) and uses less RAM, providing a rapid‑response option when latency is critical or the primary model is unavailable.

**Trade‑offs**
- The primary model is slower and heavier (≈ 1.9 GB RAM) but delivers higher accuracy, especially for noisy user‑generated content.  The fallback trades a 20 % accuracy drop for a **~3× speed gain**.

---

### Translation

**Candidate Models**
- Primary: `t5-base`
- Fallback: `t5-small`

**Why These Models Were Considered**
- **T5‑base** – a 220 M‑parameter sequence‑to‑sequence transformer that has demonstrated strong multilingual translation capabilities.
- **T5‑small** – a 60 M‑parameter variant offering lower latency and memory consumption, suitable as a backup.

**Evaluation Criteria**
- **Quality Metric:** Sentence‑level BLEU (with smoothing).
- **Latency:** average inference time per prompt.
- **Language Coverage:** performance across the languages present in the custom evaluation set.

**Comparative Results**
| Model | BLEU | Avg Latency (ms) | Role |
|---|---:|---:|---|
| `t5-base` | 0.6264 | 614.13 | Primary |
| `t5-small` | 0.5638 | 489.13 | Fallback |

**Primary Model Decision**
- `t5-base` achieved a **higher BLEU score** (0.6264 > 0.5638) while keeping latency within the acceptable range for the service.  It is therefore the **best‑performing candidate in this evaluation**.

**Fallback Model Decision**
- `t5-small` provides faster inference and a smaller memory footprint (≈ 1.5 GB vs. ≈ 1.2 GB for `t5-base`), making it a sensible fallback for high‑throughput scenarios.

**Trade‑offs**
- The primary model delivers superior translation quality, especially for the **validated languages** (French, German, Romanian).  The fallback sacrifices some BLEU but gains speed and lower RAM usage.

---

## Model‑Suitability: Translation Language Investigation

The current evaluation set contains translations for **French, German, Romanian, and Spanish** (see `assets/evaluation_dataset.json`).  Empirical BLEU scores indicate that the **primary T5‑base model consistently surpasses the fallback across French, German, and Romanian** (average BLEU > 0.60).  For **Spanish**, the BLEU score is noticeably lower, and the fallback does not close the gap sufficiently.

> **Conclusion:** The system officially supports **French, German, and Romanian** as *validated* target languages.  **Spanish, Hindi, and Telugu** are omitted from the supported scope because the evaluated configuration did **not reliably meet quality thresholds** for those languages.

---

## Limitations

- The benchmark dataset contains **only 5 samples per task** (15 total), which limits statistical confidence.
- BLEU, ROUGE‑L, and Accuracy are **task‑specific proxies** and have known shortcomings (e.g., BLEU does not capture lexical variety).
- Reported latency values are measured on the development hardware used for the benchmark; real‑world latency will vary with deployment environment.
- Results reflect **only the models listed in `config/model_registry.yaml`**; additional candidates were not evaluated.
- Model behavior can change with prompt phrasing; the numbers represent the exact prompts from `evaluation_dataset.json`.

---

*This documentation is intended to provide transparent insight into the model‑selection process for the NLP System and to guide future evaluation extensions.*
