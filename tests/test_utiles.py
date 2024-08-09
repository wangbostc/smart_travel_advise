import pytest
from adviser.utils import (
    extract_content_from_text,
    detect_injection,
    INJECTION_PATTERNS,
)


@pytest.mark.parametrize(
    "text, start_word, end_word, extraction_length, expected",
    [
        ("This is a test", "that", "test", 3, ""),
        ("This is a test", "is", "", 4, "is i"),
        ("This is a test", " is", "that", 4, " is "),
        ("This is a test", "This", "a", 4, "This is "),
    ],
)
def test_extract_content_from_text(
    text: str, start_word: str, end_word: str, extraction_length: str, expected: str
):
    assert (
        extract_content_from_text(text, start_word, end_word, extraction_length)
        == expected
    )


@pytest.mark.parametrize(
    "query",
    INJECTION_PATTERNS,
)
def test_detect_injection(query: str):
    assert detect_injection(query) == True
