import pytest
from unittest.mock import patch, Mock
from langchain_core.documents.base import Document
from adviser.adviser_support_info_retriver import (
    get_url_for_travel_advice,
    transform_html_content,
    load_from_url,
    Country,
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


@pytest.mark.parametrize(
    "country",
    [
        Country(name="china", region="asia"),
        Country(name="france", region="europe"),
        Country(name="australia", region="pacfic"),
    ],
)
def test_get_url_for_travel_advice(country: Country):
    expected_url = (
        f"https://www.smartraveller.gov.au/destinations/{country.region}/{country.name}"
    )
    result = get_url_for_travel_advice(country)
    assert result == expected_url


@pytest.mark.parametrize(
    "country",
    [
        Country(name="", region="asia"),
        Country(name="france", region=""),
        Country(name="", region=""),
    ],
)
def test_get_url_for_travel_advice_exception_handling(country: Country):
    with pytest.raises(ValueError, match="Please provide the region and country"):
        get_url_for_travel_advice(country)


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
