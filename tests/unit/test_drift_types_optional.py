import sys
from unittest.mock import patch

import pytest


def test_text_similarity_missing_dependency():
    """Test that text_similarity raises ImportError when sentence-transformers is not available."""
    from promptdrifter.drift_types import text_similarity

    with patch.dict(sys.modules, {"sentence_transformers": None}):
        with pytest.raises(ImportError, match="text_similarity requires the 'sentence-transformers' package"):
            text_similarity("hello world", "hello world")


def test_text_similarity_helpful_error_message():
    """Test that the error message includes installation instructions."""
    from promptdrifter.drift_types import text_similarity

    with patch.dict(sys.modules, {"sentence_transformers": None}):
        with pytest.raises(ImportError) as exc_info:
            text_similarity("hello", "world")

        error_message = str(exc_info.value)
        assert "pip install 'promptdrifter[similarity]'" in error_message
        assert "pip install sentence-transformers" in error_message
