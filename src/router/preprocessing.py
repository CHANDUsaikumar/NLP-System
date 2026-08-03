"""Pre-processing module extracting structural, syntactic, and linguistic features from user input."""

import re
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Pattern, Final


class DocumentSize(str, Enum):
    """Enumeration of input document size categories."""
    SHORT = "SHORT"      # < 30 words
    MEDIUM = "MEDIUM"    # 30 - 150 words
    LONG = "LONG"        # > 150 words


@dataclass
class PreprocessedInput:
    """Structured data holding pre-processed text features."""
    raw_text: str
    cleaned_text: str
    char_count: int
    word_count: int
    sentence_count: int
    has_question_mark: bool
    question_mark_count: int
    starts_with_interrogative: bool
    interrogative_word: str
    starts_with_imperative: bool
    imperative_verb: str
    contains_sentiment_words: bool
    contains_translation_phrase: bool
    contains_ner_directive: bool
    translation_target: str = ""
    document_size: DocumentSize = DocumentSize.SHORT
    detected_sentiment_words: List[str] = field(default_factory=list)


class TextPreprocessor:
    """Cleans text and extracts syntactic and heuristic indicators for task routing."""

    INTERROGATIVE_WORDS: Final[List[str]] = [
        "what", "why", "when", "who", "where", "which", "how",
        "can", "could", "would", "should", "is", "are", "do", "does", "did"
    ]

    IMPERATIVE_GENERATION_VERBS: Final[List[str]] = [
        "write", "generate", "continue", "compose", "create", "tell", "draft", "story", "poem", "essay"
    ]

    SENTIMENT_INDICATORS: Final[List[str]] = [
        "great", "terrible", "awesome", "awful", "good", "bad", "love", "hate",
        "recommend", "worst", "best", "excellent", "poor", "horrible", "fantastic",
        "disappointed", "amazing", "wonderful", "enjoyed", "loved", "hated", "like", "dislike"
    ]

    # Pre-compiled class-level Regex patterns for optimal performance
    WHITESPACE_REGEX: Final[Pattern] = re.compile(r"\s+")
    SENTENCE_SPLIT_REGEX: Final[Pattern] = re.compile(r"[.!?]+")

    TRANSLATION_PATTERNS: Final[List[Pattern]] = [
        re.compile(p) for p in [
            r"\btranslate\b",
            r"\bconvert to\b",
            r"\btranslate into\b",
            r"\bin (french|spanish|german|chinese|japanese|italian|hindi|arabic|russian|portuguese)\b",
            r"\bto (french|spanish|german|chinese|japanese|italian|hindi|arabic|russian|portuguese)\b"
        ]
    ]

    NER_DIRECTIVE_PATTERNS: Final[List[Pattern]] = [
        re.compile(p) for p in [
            r"\b(find|extract|identify)\s+(entities|names|organizations|locations|people|places)\b",
            r"\bner\b",
            r"\bnamed entity\b",
            r"\bextract names\b",
            r"\bidentify organizations\b"
        ]
    ]

    def preprocess(self, text: str) -> PreprocessedInput:
        """Processes raw text and returns structured PreprocessedInput dataclass.

        Args:
            text (str): Raw input prompt text provided by user.

        Returns:
            PreprocessedInput: Dataclass containing extracted features, counts, and indicators.
        """
        cleaned = self.WHITESPACE_REGEX.sub(" ", text.strip())
        words = [w.strip(".,!?;:\"'") for w in cleaned.split() if w.strip(".,!?;:\"'")]
        word_count = len(words)
        char_count = len(cleaned)

        sentences = [s.strip() for s in self.SENTENCE_SPLIT_REGEX.split(cleaned) if s.strip()]
        sentence_count = max(len(sentences), 1 if char_count > 0 else 0)

        has_question_mark = "?" in cleaned
        question_mark_count = cleaned.count("?")

        first_word = words[0].lower() if words else ""
        starts_with_interrogative = first_word in self.INTERROGATIVE_WORDS
        interrogative_word = first_word if starts_with_interrogative else ""

        starts_with_imperative = first_word in self.IMPERATIVE_GENERATION_VERBS
        imperative_verb = first_word if starts_with_imperative else ""

        found_sentiment_words = [w.lower() for w in words if w.lower() in self.SENTIMENT_INDICATORS]
        contains_sentiment_words = len(found_sentiment_words) > 0

        lower_text = cleaned.lower()
        contains_translation_phrase = False
        translation_target = ""
        for pattern in self.TRANSLATION_PATTERNS:
            match = pattern.search(lower_text)
            if match:
                contains_translation_phrase = True
                translation_target = match.group(0)
                break

        contains_ner_directive = any(pat.search(lower_text) for pat in self.NER_DIRECTIVE_PATTERNS)

        if word_count < 30:
            doc_size = DocumentSize.SHORT
        elif word_count <= 150:
            doc_size = DocumentSize.MEDIUM
        else:
            doc_size = DocumentSize.LONG

        return PreprocessedInput(
            raw_text=text,
            cleaned_text=cleaned,
            char_count=char_count,
            word_count=word_count,
            sentence_count=sentence_count,
            has_question_mark=has_question_mark,
            question_mark_count=question_mark_count,
            starts_with_interrogative=starts_with_interrogative,
            interrogative_word=interrogative_word,
            starts_with_imperative=starts_with_imperative,
            imperative_verb=imperative_verb,
            contains_sentiment_words=contains_sentiment_words,
            contains_translation_phrase=contains_translation_phrase,
            contains_ner_directive=contains_ner_directive,
            translation_target=translation_target,
            document_size=doc_size,
            detected_sentiment_words=found_sentiment_words
        )
