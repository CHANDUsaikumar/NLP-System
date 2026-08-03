"""Unit tests for TextPreprocessor and feature extraction."""

import pytest
from src.router.preprocessing import TextPreprocessor, PreprocessedInput


def test_preprocessor_question_detection():
    preprocessor = TextPreprocessor()
    res = preprocessor.preprocess("What is the speed of light?")

    assert res.has_question_mark is True
    assert res.starts_with_interrogative is True
    assert res.interrogative_word == "what"
    assert res.word_count == 6


def test_preprocessor_imperative_detection():
    preprocessor = TextPreprocessor()
    res = preprocessor.preprocess("Write a poem about quantum computers.")

    assert res.starts_with_imperative is True
    assert res.imperative_verb == "write"


def test_preprocessor_translation_detection():
    preprocessor = TextPreprocessor()
    res = preprocessor.preprocess("Translate this article into French.")

    assert res.contains_translation_phrase is True


def test_preprocessor_ner_detection():
    preprocessor = TextPreprocessor()
    res = preprocessor.preprocess("Extract names and organizations from the text.")

    assert res.contains_ner_directive is True


def test_preprocessor_doc_size():
    preprocessor = TextPreprocessor()

    short_text = "Hello world."
    long_text = "word " * 160

    assert preprocessor.preprocess(short_text).document_size == "SHORT"
    assert preprocessor.preprocess(long_text).document_size == "LONG"
