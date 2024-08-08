import pytest
from unittest.mock import patch, MagicMock, Mock
from langchain_core.documents.base import Document
from adviser.adviser_support_info_retriver import (
    get_tool_calling_model,
    get_url_for_travel_advice,
    transform_html_content,
    load_from_url,
)


@pytest.fixture
def html_content_document():
    return [
        Document(
            metadata={
                "source": "http://test.com",
                "title": "advice for test",
                "description": "It is safe to test",
            },
            page_content="<html><p>test html content</p></html>",
        )
    ]


@pytest.fixture
def text_document():
    return Document(
        metadata={
            "source": "http://test.com",
            "title": "advice for test",
            "description": "It is safe to test",
        },
        page_content="test html content\n\n",
    )


@patch("adviser.adviser_support_info_retriver.StructuredTool")
@patch("adviser.adviser_support_info_retriver.BaseChatModel")
def test_get_tool_calling_model_with_bind_functions(
    mock_chat_model: Mock, mock_structured_tool: Mock
):
    # Create a mock chat model with bind_functions
    mock_chat_model.bind_functions = MagicMock()
    mock_chat_model.bind_tools.return_value = "Mocked binding"
    result = get_tool_calling_model(mock_chat_model, [mock_structured_tool])

    mock_chat_model.bind_tools.assert_called_once_with(
        tools=[mock_structured_tool], tool_choice="required"
    )
    assert result == "Mocked binding"


@patch("adviser.adviser_support_info_retriver.StructuredTool")
@patch("adviser.adviser_support_info_retriver.BaseChatModel")
def test_get_tool_calling_model_without_bind_functions(
    mock_chat_model: Mock, mock_structured_tool: Mock
):
    del mock_chat_model.bind_functions
    with pytest.raises(
        ValueError, match="The provided chat model does not support binding functions."
    ):
        get_tool_calling_model(mock_chat_model, [mock_structured_tool])


@pytest.mark.parametrize(
    "region, country",
    [
        ("asia", "china"),
        ("europe", "france"),
        ("pacfic", "australia"),
    ],
)
def test_get_url_for_travel_advice(region: str, country: str):
    expected_url = f"https://www.smartraveller.gov.au/destinations/{region}/{country}"
    result = get_url_for_travel_advice({"region": region, "country": country})
    assert result == expected_url


@pytest.mark.parametrize(
    "region, country",
    [
        ("", "china"),
        ("europe", ""),
        ("", ""),
    ],
)
def test_get_url_for_travel_advice_exception_handling(region: str, country: str):
    with pytest.raises(ValueError, match="Please provide the region and country"):
        get_url_for_travel_advice({"region": region, "country": country})


@patch("langchain_community.document_loaders.WebBaseLoader.load")
def test_load_from_url_success(mock_web_load: Mock, html_content_document: Document):
    url = "http://test.com"
    mock_web_load.return_value = html_content_document
    result = load_from_url(url)
    assert result == html_content_document


@patch("langchain_community.document_loaders.WebBaseLoader.load")
def test_load_from_url_exception_handling(mock_web_load: Mock):
    url = "http://test.com"
    mock_web_load.side_effect = Exception("Loader error")
    with pytest.raises(Exception, match="Loader error"):
        load_from_url(url)


def test_transform_html_content(
    html_content_document: Document, text_document: Document
):
    text_content_doc = transform_html_content(html_content_document)
    assert text_content_doc == text_document
