"""Evaluation Metrics engine computing ROUGE-1, ROUGE-2, ROUGE-L, BLEU, and BERTScore semantic similarity."""

import functools
from typing import Dict, Any, Optional
from src.utils.logger import logger
from src.utils.exceptions import EvaluationError

try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False


@functools.lru_cache(maxsize=1)
def _get_cached_bert_scorer():
    """Lazily initializes and caches BERTScorer model object in memory."""
    from bert_score import BERTScorer
    return BERTScorer(lang="en", rescale_with_baseline=True)


class EvaluationMetrics:
    """Computes quantitative text evaluation metrics comparing generated output against reference text."""

    @staticmethod
    def compute_rouge(candidate: str, reference: str) -> Dict[str, float]:
        """Computes ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

        Args:
            candidate (str): Generated candidate output text.
            reference (str): Ground-truth reference text.

        Returns:
            Dict[str, float]: Dictionary of ROUGE F1 metrics.
        """
        if not candidate or not reference:
            return {"rouge1_f1": 0.0, "rouge2_f1": 0.0, "rougeL_f1": 0.0}
            
        try:
            if HAS_ROUGE:
                scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
                scores = scorer.score(reference, candidate)
                return {
                    "rouge1_f1": round(scores['rouge1'].fmeasure, 4),
                    "rouge2_f1": round(scores['rouge2'].fmeasure, 4),
                    "rougeL_f1": round(scores['rougeL'].fmeasure, 4)
                }
            else:
                # Token-level overlap fallback calculation if rouge-score library is not installed
                cand_tokens = set(candidate.lower().split())
                ref_tokens = set(reference.lower().split())
                overlap = cand_tokens.intersection(ref_tokens)
                if not cand_tokens or not ref_tokens:
                    f1 = 0.0
                else:
                    prec = len(overlap) / len(cand_tokens)
                    rec = len(overlap) / len(ref_tokens)
                    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
                return {
                    "rouge1_f1": round(f1, 4),
                    "rouge2_f1": round(f1, 4),
                    "rougeL_f1": round(f1, 4)
                }
        except Exception as e:
            logger.error(f"Error computing ROUGE metrics: {e}")
            raise EvaluationError(f"ROUGE evaluation failed: {e}") from e

    @staticmethod
    def compute_bert_score(candidate: str, reference: str) -> Dict[str, float]:
        """Computes BERTScore precision, recall, and F1 contextual embeddings similarity.

        Args:
            candidate (str): Generated text candidate.
            reference (str): Ground-truth reference text.

        Returns:
            Dict[str, float]: Dictionary containing precision, recall, and f1 bert_score.
        """
        try:
            scorer = _get_cached_bert_scorer()
            P, R, F1 = scorer.score([candidate], [reference])
            return {
                "bert_score_precision": round(float(P[0]), 4),
                "bert_score_recall": round(float(R[0]), 4),
                "bert_score_f1": round(float(F1[0]), 4)
            }
        except Exception as e:
            logger.warning(f"BERTScore computation skipped or unavailable ({e}). Returning 0.0.")
            return {
                "bert_score_f1": 0.0,
                "note": "BERTScore model computation skipped or unavailable."
            }

    @staticmethod
    def compute_bleu(candidate: str, reference: str) -> Dict[str, float]:
        """Computes sentence BLEU score.

        Args:
            candidate (str): Candidate string.
            reference (str): Reference string.

        Returns:
            Dict[str, float]: BLEU score dictionary.
        """
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            ref_tokens = [reference.lower().split()]
            cand_tokens = candidate.lower().split()
            smooth = SmoothingFunction().method1
            bleu = sentence_bleu(ref_tokens, cand_tokens, smoothing_function=smooth)
            return {"bleu_score": round(float(bleu), 4)}
        except Exception as e:
            logger.warning(f"BLEU computation failed: {e}")
            return {"bleu_score": 0.0}

    @staticmethod
    def compute_sentiment_metrics(y_true: list, y_pred: list) -> Dict[str, float]:
        """Computes Accuracy, Precision, Recall, and F1-score for sentiment classification."""
        try:
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support
            acc = accuracy_score(y_true, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
            return {
                "accuracy": round(float(acc), 4),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1_score": round(float(f1), 4)
            }
        except Exception as e:
            logger.error(f"Sentiment metrics calculation error: {e}")
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}

    @staticmethod
    def compute_qa_metrics(candidate: str, reference: str) -> Dict[str, float]:
        """Computes Exact Match (EM) and Token-level F1 score for Question Answering."""
        c_clean = candidate.strip().lower()
        r_clean = reference.strip().lower()
        exact_match = 1.0 if c_clean == r_clean else 0.0

        c_tokens = c_clean.split()
        r_tokens = r_clean.split()
        common = set(c_tokens).intersection(set(r_tokens))
        
        if not c_tokens or not r_tokens:
            f1 = 0.0
        else:
            prec = len(common) / len(c_tokens)
            rec = len(common) / len(r_tokens)
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        return {
            "exact_match": exact_match,
            "f1_score": round(f1, 4)
        }

    @classmethod
    def compute_all(cls, candidate: str, reference: str) -> Dict[str, Any]:
        """Runs quantitative evaluation metrics suite.

        Args:
            candidate (str): Candidate generated output.
            reference (str): Reference text.

        Returns:
            Dict[str, Any]: Consolidated metrics dict.
        """
        if not reference or not reference.strip():
            return {}

        results = {}
        results.update(cls.compute_rouge(candidate, reference))
        results.update(cls.compute_bleu(candidate, reference))
        results.update(cls.compute_bert_score(candidate, reference))
        return results
