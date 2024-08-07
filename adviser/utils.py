from typing import Optional


def extract_content_from_text(
    text: str,
    start_word: str,
    end_word: Optional[str] = "",
    extraction_length: int = 1000,
) -> str:
    """Extract the content from the smart traveler document starts with
    the search word with lenghth of extraction_length from text. Be careful
    that the search_word should be case sensitive, and only the first
    match will be returned.
    """
    start_position = text.find(start_word)

    if start_position == -1:
        return ""

    if end_word == "":
        return text[start_position : start_position + extraction_length]

    end_position = text.find(end_word, start_position)

    if end_position == -1:
        end_position = start_position + extraction_length

    return text[start_position:end_position]
