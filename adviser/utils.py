from typing import Optional


def extract_content_from_text(
    text: str,
    start_word: str,
    end_word: Optional[str] = "",
    extraction_length: int = 1000,
) -> str:
    """Extracts the content from a text document starting and ending with the specified search word.
    
    Args:
        text (str): The text to extract content from.
        start_word (str): The word to start the extraction from. The search is case sensitive.
        end_word (str, optional): The word to end the extraction at. If not provided, the extraction will be of length extraction_length. Defaults to "".
        extraction_length (int, optional): The length of the extraction if end_word is not provided. Defaults to 1000.
    
    Returns:
        str: The extracted content from the text.
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
