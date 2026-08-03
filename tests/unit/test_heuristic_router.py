"""Unit tests for HeuristicRouter rule matching."""

import pytest
from src.router.preprocessing import TextPreprocessor
from src.router.heuristic_router import HeuristicRouter


def test_heuristic_qa_match():
    prep = TextPreprocessor().preprocess("Why do stars shine?")
    router = HeuristicRouter()
    match = router.evaluate(prep)

    assert match is not None
    assert match.task_key == "question_answering"
    assert match.confidence >= 0.90


def test_heuristic_text_gen_match():
    prep = TextPreprocessor().preprocess("Generate a story about space exploration.")
    router = HeuristicRouter()
    match = router.evaluate(prep)

    assert match is not None
    assert match.task_key == "text_generation"


def test_heuristic_translation_match():
    prep = TextPreprocessor().preprocess("Translate this text to Spanish.")
    router = HeuristicRouter()
    match = router.evaluate(prep)

    assert match is not None
    assert match.task_key == "translation"


def test_heuristic_ner_match():
    prep = TextPreprocessor().preprocess("Find entities in the following report.")
    router = HeuristicRouter()
    match = router.evaluate(prep)

    assert match is not None
    assert match.task_key == "named_entity_recognition"


def test_heuristic_summarization_long_doc():
    long_doc = "Artificial intelligence models process natural language. " * 30
    prep = TextPreprocessor().preprocess(long_doc)
    router = HeuristicRouter()
    match = router.evaluate(prep)

    assert match is not None
    assert match.task_key == "summarization"


def test_heuristic_inconclusive():
    prep = TextPreprocessor().preprocess("Global climate patterns are changing.")
    router = HeuristicRouter()
    match = router.evaluate(prep)

    assert match is None
