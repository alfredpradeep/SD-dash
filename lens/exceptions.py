class LEAEError(Exception):
    """Base exception for LENS engine errors."""
    pass


class UnsupportedLanguageError(LEAEError):
    """Raised when a language code is not supported."""
    def __init__(self, language: str):
        super().__init__(f"Unsupported language: '{language}'. Check SUPPORTED_LANGUAGES.")
        self.language = language


class UnsupportedModelError(LEAEError):
    """Raised when a model name is not recognised."""
    def __init__(self, model: str):
        super().__init__(f"Unsupported model: '{model}'.")
        self.model = model


class StorageError(LEAEError):
    """Raised when a storage operation fails."""
    pass


class LanguageDetectionError(LEAEError):
    """Raised when language detection fails or returns low confidence."""
    pass


class EntropyCalculationError(LEAEError):
    """Raised when entropy calculation encounters an unrecoverable error."""
    pass


class CompressBridgeError(LEAEError):
    """Raised when the COMPRESS engine bridge call fails."""
    pass
