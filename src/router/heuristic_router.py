"""Rule-based Heuristic Router evaluating weighted linguistic scoring prior to zero-shot classification."""

import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Pattern, Final, Tuple
from src.router.preprocessing import PreprocessedInput, DocumentSize
from config.settings import TaskCategory


@dataclass
class HeuristicMatch:
    """Represents a successful weighted heuristic rule match."""
    task_key: str
    confidence: float
    rule_name: str
    description: str
    score: int


class HeuristicRouter:
    """Evaluates rule-based heuristics on preprocessed text features using a weighted scoring model."""

    SUMMARIZATION_PATTERNS: Final[List[Pattern]] = [
        re.compile(r"\b(summarize|summary|tldr|tl;dr|abstract|condense|shorten|overview)\b", re.IGNORECASE)
    ]

    SENTIMENT_PATTERNS: Final[List[Pattern]] = [
        re.compile(r"\b(sentiment|feeling|emotion|opinion|rating|review|feedback)\b", re.IGNORECASE)
    ]

    MINIMUM_SCORE_THRESHOLD: Final[int] = 4

    def evaluate(self, processed: PreprocessedInput) -> Optional[HeuristicMatch]:
        """Evaluates weighted scoring across candidate task rules.

        Weighting Rules:
        - Question Answering: Question Mark (+4), Interrogative Word (+3)
        - Summarization: Explicit Keyword (+5), Long Doc (+3), Multiple Sentences (+2)
        - Sentiment Analysis: Explicit Keyword (+5), Opinion Words (+4)
        - Translation: Translation Phrase (+5)
        - Text Generation: Imperative Verb (+5)
        - Named Entity Recognition: NER Directive (+5)

        Args:
            processed (PreprocessedInput): Pre-processed text feature object.

        Returns:
            Optional[HeuristicMatch]: Match object if max score >= MINIMUM_SCORE_THRESHOLD, else None.
        """
        scores: Dict[str, int] = {task.value: 0 for task in TaskCategory}
        reasons: Dict[str, List[str]] = {task.value: [] for task in TaskCategory}
        lower_text = processed.cleaned_text.lower()

        # 1. Translation Directive
        if processed.contains_translation_phrase:
            scores[TaskCategory.TRANSLATION.value] += 5
            reasons[TaskCategory.TRANSLATION.value].append(f"Translation Phrase ('{processed.translation_target}') [+5]")

        # 2. NER Directive
        if processed.contains_ner_directive:
            scores[TaskCategory.NAMED_ENTITY_RECOGNITION.value] += 5
            reasons[TaskCategory.NAMED_ENTITY_RECOGNITION.value].append("NER Directive [+5]")

        # 3. Text Generation Directive
        if processed.starts_with_imperative:
            scores[TaskCategory.TEXT_GENERATION.value] += 5
            reasons[TaskCategory.TEXT_GENERATION.value].append(f"Generation Verb ('{processed.imperative_verb.capitalize()}') [+5]")

        # 4. Question Answering
        if processed.has_question_mark:
            scores[TaskCategory.QUESTION_ANSWERING.value] += 4
            reasons[TaskCategory.QUESTION_ANSWERING.value].append("Question Mark [+4]")

        if processed.starts_with_interrogative:
            scores[TaskCategory.QUESTION_ANSWERING.value] += 3
            reasons[TaskCategory.QUESTION_ANSWERING.value].append(f"Interrogative Word ('{processed.interrogative_word.capitalize()}') [+3]")

        # 5. Summarization
        for pattern in self.SUMMARIZATION_PATTERNS:
            if pattern.search(lower_text):
                scores[TaskCategory.SUMMARIZATION.value] += 5
                reasons[TaskCategory.SUMMARIZATION.value].append("Summarization Keyword [+5]")
                break

        if processed.document_size == DocumentSize.LONG and not processed.has_question_mark:
            scores[TaskCategory.SUMMARIZATION.value] += 3
            reasons[TaskCategory.SUMMARIZATION.value].append("Long Document [+3]")

        if processed.sentence_count > 2 and not processed.has_question_mark:
            scores[TaskCategory.SUMMARIZATION.value] += 2
            reasons[TaskCategory.SUMMARIZATION.value].append("Multiple Sentences [+2]")

        # 6. Sentiment Analysis
        for pattern in self.SENTIMENT_PATTERNS:
            if pattern.search(lower_text):
                scores[TaskCategory.SENTIMENT.value] += 5
                reasons[TaskCategory.SENTIMENT.value].append("Sentiment Keyword [+5]")
                break

        if processed.contains_sentiment_words and processed.word_count < 80:
            words_str = ", ".join([f"'{w}'" for w in processed.detected_sentiment_words[:3]])
            scores[TaskCategory.SENTIMENT.value] += 4
            reasons[TaskCategory.SENTIMENT.value].append(f"Opinion Indicators ({words_str}) [+4]")

        # Select highest scoring candidate task
        top_task, top_score = max(scores.items(), key=lambda x: x[1])

        if top_score < self.MINIMUM_SCORE_THRESHOLD:
            return None

        matched_reasons = ", ".join(reasons[top_task])
        confidence = min(0.85 + (top_score * 0.02), 0.98)

        return HeuristicMatch(
            task_key=top_task,
            confidence=round(confidence, 2),
            rule_name="Weighted Score Match",
            description=f"Weighted Rules: {matched_reasons} -> Total Score: {top_score}.",
            score=top_score
        )
