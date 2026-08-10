"""Evaluation Metrics engine computing ROUGE-L, Sentiment Accuracy, and BLEU metrics."""

import re
from typing import Dict, Any, List

try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False


class EvaluationMetrics:
    """Computes quantitative quality evaluation metrics comparing generated output against reference text."""

    @staticmethod
    def compute_rouge_l(candidate: str, reference: str) -> float:
        """Computes ROUGE-L F1 score for summarization."""
        if not candidate or not reference:
            return 0.0
            
        try:
            if HAS_ROUGE:
                scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
                scores = scorer.score(reference, candidate)
                return round(float(scores['rougeL'].fmeasure), 4)
            else:
                cand_tokens = set(candidate.lower().split())
                ref_tokens = set(reference.lower().split())
                overlap = cand_tokens.intersection(ref_tokens)
                if not cand_tokens or not ref_tokens:
                    return 0.0
                prec = len(overlap) / len(cand_tokens)
                rec = len(overlap) / len(ref_tokens)
                f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
                return round(f1, 4)
        except Exception:
            return 0.0

    @staticmethod
    def compute_accuracy(y_true: List[str], y_pred: List[str]) -> float:
        """Computes classification accuracy score for sentiment analysis."""
        if not y_true or not y_pred or len(y_true) != len(y_pred):
            return 0.0

        matches = 0
        for t, p in zip(y_true, y_pred):
            t_clean = t.strip().lower()
            p_clean = p.strip().lower()
            if t_clean in p_clean or p_clean in t_clean:
                matches += 1

        return round(matches / len(y_true), 4)

    @staticmethod
    def compute_bleu(candidate: str, reference: str) -> float:
        """Computes sentence BLEU score for translation."""
        if not candidate or not reference:
            return 0.0

        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            ref_tokens = [reference.lower().split()]
            cand_tokens = candidate.lower().split()
            smooth = SmoothingFunction().method1
            bleu = sentence_bleu(ref_tokens, cand_tokens, smoothing_function=smooth)
            return round(float(bleu), 4)
        except Exception:
            # Simple fallback n-gram precision calculation if NLTK is not available
            c_words = candidate.lower().split()
            r_words = reference.lower().split()
            if not c_words or not r_words:
                return 0.0
            overlap = len(set(c_words).intersection(set(r_words)))
            return round(overlap / max(len(c_words), len(r_words)), 4)
