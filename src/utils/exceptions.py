"""Custom exception hierarchy for the Adaptive NLP System."""


class NLPSystemError(Exception):
    """Base exception for all domain errors within the NLP system."""
    pass


class InputValidationError(NLPSystemError):
    """Raised when incoming prompt or request payload fails validation."""
    pass


class RoutingError(NLPSystemError):
    """Raised when intent classification or model selection fails."""
    pass


class ModelLoadError(NLPSystemError):
    """Raised when a transformer model fails to load into memory or device."""
    pass


class InferenceError(NLPSystemError):
    """Raised when execution of a model pipeline fails during inference."""
    pass


class EvaluationError(NLPSystemError):
    """Raised when score evaluation (ROUGE / BERTScore) fails."""
    pass
