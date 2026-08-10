# Evaluation Methodology

This document describes the **complete evaluation pipeline** used to benchmark the models in the NLP System.  The steps are implemented in `src/evaluation/model_evaluator.py` and orchestrated by the CLI in `evaluate.py`.

---

## 1. Dataset Selection
- **Source:** `assets/evaluation_dataset.json`
- **Content:** 5 summarization prompts, 5 sentiment reviews, and 5 translation queries (French, German, Romanian, Spanish, Hindi).
- **Rationale:** The dataset is a curated **custom evaluation set** that mirrors the types of inputs the system receives (technical articles, product reviews, multilingual sentences).  It is deliberately **small** to enable rapid benchmarking while still covering the three core tasks.

## 2. Data Validation
- The JSON file is loaded and each entry is validated against a simple schema (presence of `id`, `prompt`, `target_task`, and `reference_text`).
- Invalid entries raise an exception and abort the benchmark, ensuring only well‑formed samples are used.

## 3. Preprocessing
- **Summarization & Sentiment:** Whitespace trimming of the prompt; for sentiment, the reference label is lower‑cased.
- **Translation:** Tokenization and NLTK BLEU sentence‑level smoothing are applied in `src/evaluation/metrics.py`.
- All tasks share a common step of converting the prompt string into the format expected by the model pipelines.

## 4. Model Inference
- For each **task** (`summarization`, `sentiment`, `translation`) the primary and fallback model names are read from `config/model_registry.yaml`.
- `src.models.model_manager.ModelManager` builds an inference pipeline for the given model name.
- Each sample is passed to the pipeline via `pipeline.run(prompt)`, which returns the generated text and the measured inference latency (ms).

## 5. Task‑specific Metric
| Task | Metric | How Computed |
|------|--------|--------------|
| Summarization | **ROUGE‑L** (F1) | `EvaluationMetrics.compute_rouge_l(generated, reference)` using the `rouge_score` library.
| Sentiment | **Accuracy** | Exact string match between predicted label and `reference_text`.
| Translation | **BLEU** | Sentence‑level BLEU with smoothing, via `EvaluationMetrics.compute_bleu(generated, reference)`.

## Metric Selection and Calculation

### Summarization — ROUGE‑L

- **Why selected:** Summarization generates free‑form text. ROUGE‑L measures how much of the reference summary’s longest common subsequence (LCS) is retained, which captures ordering and content overlap better than an exact‑match metric.
- **What it measures:** Precision = LCS length / candidate length, Recall = LCS length / reference length, and the F‑score (harmonic mean) combines them:
  \[\text{ROUGE‑L F1} = \frac{2 \times P \times R}{P + R}\]
- **Implementation:** `EvaluationMetrics.compute_rouge_l(candidate, reference)`
  - If the `rouge_score` library is available, it creates a `RougeScorer(['rougeL'], use_stemmer=True)` and returns the `fmeasure` rounded to 4 decimals.
  - Otherwise a fallback splits both strings into lower‑cased tokens, computes token overlap, derives precision and recall as above, and returns the rounded F1.
- **Interpretation:** Values range from 0.0 to 1.0; higher scores indicate a summary closer to the reference. Typical useful ranges are task‑specific; the metric is not an absolute quality guarantee.
- **Limitations:** ROUGE‑L accounts only for lexical overlap; it does not capture semantic equivalence, paraphrasing, or factual correctness. The fallback implementation is a coarse approximation.

### Sentiment Analysis — Accuracy

- **Why selected:** Sentiment analysis yields a discrete class label. Accuracy directly reports the proportion of correct predictions.
- **What it measures:** \[\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}\]
- **Implementation:** `EvaluationMetrics.compute_accuracy(y_true, y_pred)`
  - Normalises both ground‑truth and predicted strings (trim, lower‑case).
  - Counts a match when one string is a substring of the other (allowing minor format variations).
  - Returns the ratio rounded to 4 decimals.
- **Interpretation:** 0 → all predictions wrong, 1 → perfect classification. Provides a quick overall view of model correctness.
- **Limitations:** Accuracy can be misleading on imbalanced datasets (dominant class inflates score). The substring‑based match is tolerant but may over‑state performance.

### Translation — Sentence BLEU

- **Why selected:** Translation outputs free‑form sentences that can be compared against a reference translation using n‑gram overlap.
- **What it measures:** Sentence‑level BLEU aggregates modified n‑gram precisions (1‑gram to 4‑gram) with a brevity penalty (BP):
  \[\text{BLEU} = \text{BP} \times \exp\left( \sum_{n=1}^4 w_n \log p_n \right)\]
  where \(p_n\) is the precision for n‑grams and \(w_n = \frac{1}{4}\).
- **Implementation:** `EvaluationMetrics.compute_bleu(candidate, reference)`
  - Uses NLTK’s `sentence_bleu` with `SmoothingFunction().method1` for stability, rounding to 4 decimals.
  - If NLTK is unavailable, falls back to a simple overlap‑based precision: \( \frac{|cand\cap\,ref|}{\max(|cand|,|ref|)}\).
- **Interpretation:** Scores range from 0.0 to 1.0; higher indicates greater n‑gram overlap with the reference. Results are typically averaged across samples ("Average Sentence BLEU").
- **Limitations:** BLEU captures only surface n‑gram similarity; it does not account for meaning, synonymy, or valid re‑phrasings. Sentence‑level BLEU can be noisy for very short sentences.

### Latency — All Tasks

- **Why tracked:** Latency reflects the inference speed, a practical performance criterion alongside quality.
- **What it measures:** Inference time in milliseconds for each model call, as returned by `pipeline.run(prompt)`.
- **Implementation:** `ModelEvaluator.evaluate_model` appends each latency to a list, then computes the average:
  \[\text{Avg Latency} = \frac{\sum \text{latency}_i}{N}\]
- **Interpretation:** Lower values denote faster responses. Latency complements quality metrics to guide primary vs. fallback model choices.
- **Limitations:** Measurements depend on the hardware (CPU/GPU) and runtime environment; they are not portable across deployment settings and do not capture throughput or concurrent load.

## 6. Latency Measurement
- The latency for each inference call is captured inside the pipeline wrapper and aggregated per model.
- Average latency (`avg_latency_ms`) is reported in `logs/benchmark_results.json`.

## 7. Model Comparison
- For each task we compute **average quality score** and **average latency** across all samples.
- The model with the **higher quality score** becomes the **Primary** model; the other is recorded as **Fallback** (as defined in `model_registry.yaml`).

## 8. Primary / Fallback Selection
- The configuration in `model_registry.yaml` explicitly lists `model_name` (primary) and `fallback_model` for each task.
- The benchmark validates that the primary model indeed outperforms the fallback on the chosen quality metric.  If a fallback were to outperform, the system would still retain the configured primary model, but the documentation would note the discrepancy.

---

## Limitations of the Evaluation
- **Dataset Size:** Only 5 samples per task; results are not statistically robust.
- **Metric Coverage:** BLEU and ROUGE‑L capture n‑gram overlap but not semantic adequacy. Accuracy for sentiment does not reflect confidence or class imbalance.
- **Hardware Dependency:** Latency numbers depend on the hardware used for the benchmark (CPU/GPU configuration) and may differ in production.
- **Scope:** Only the models listed in `model_registry.yaml` are evaluated; additional candidates would require extending the registry and rerunning the benchmark.
- **Language Coverage:** Translation evaluation currently includes French, German, Romanian, and Spanish prompts; Hindi and Telugu are omitted because the evaluated models did not meet quality expectations.

---

*The evaluation pipeline is fully reproducible: running `python evaluate.py` will re‑generate the benchmark results and the JSON report.*
