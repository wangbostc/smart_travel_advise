from unittest.mock import patch
import pytest
from typing import Dict
from langchain_core.documents.base import Document

from adviser.advise_model import (
    TravelAdviceInput,
    get_required_prompt_field,
    get_required_prompt_fields,
    create_prompt_template_for_travel_advice,
    create_prompt_for_travel_advice_response,
)


# Mock data for testing
class MockDocument:
    def __init__(self, metadata, page_content):
        self.metadata = metadata
        self.page_content = page_content


@pytest.fixture
def sample_input_variables() -> Dict[str, TravelAdviceInput]:
    return {
        "title": TravelAdviceInput(name="title", html_section="metadata"),
        "description": TravelAdviceInput(name="description", html_section="metadata"),
        "test_message": TravelAdviceInput(
            name="test_message",
            html_section="page_content",
            html_section_start="Test Message",
            html_section_end="End Test",
            html_section_length=800,
        ),
        "query": TravelAdviceInput(name="query", html_section="query"),
    }


@pytest.fixture
def sample_doc() -> Document:
    return MockDocument(
        metadata={"title": "Sample Title", "description": "Sample Description"},
        page_content="Test Message: Some content here... End Test\n Some more content...",
    )


def test_define_information_input_variables(sample_input_variables):
    input_vars = sample_input_variables
    assert isinstance(input_vars, dict)
    assert list(input_vars.keys()) == ["title", "description", "test_message", "query"]
    assert all([isinstance(vals, TravelAdviceInput) for vals in input_vars.values()])


def test_get_required_prompt_field_query(sample_doc):
    input_var = TravelAdviceInput(name="query", html_section="query")
    result = get_required_prompt_field(input_var, sample_doc, "test query")
    assert result == "test query"


def test_get_required_prompt_field_metadata(sample_doc):
    input_var = TravelAdviceInput(name="title", html_section="metadata")
    result = get_required_prompt_field(input_var, sample_doc, "test query")
    assert result == "Sample Title"


def test_get_required_prompt_field_page_content(sample_doc):
    input_var = TravelAdviceInput(
        name="test_message",
        html_section="page_content",
        html_section_start="Test Message",
        html_section_end="End Test",
        html_section_length=50,
    )
    result = get_required_prompt_field(input_var, sample_doc, "test query")
    assert "Some content here..." in result


@patch("adviser.advise_model.define_information_input_variables")
def test_get_required_prompt_fields(
    mock_input_variables, sample_input_variables, sample_doc
):
    query = "Good to go?"
    mock_input_variables.return_value = sample_input_variables
    result = get_required_prompt_fields(sample_doc, query)
    assert list(result.keys()) == ["title", "description", "test_message", "query"]
    assert result["query"] == query
    assert result["title"] == "Sample Title"
    assert result["description"] == "Sample Description"
    assert "Some content here..." in result["test_message"]


def test_create_prompt_template_for_travel_advice():
    prompt_template = create_prompt_template_for_travel_advice()
    assert prompt_template is not None
    assert "{query}" in prompt_template.template


def test_create_prompt_for_travel_advice_response(sample_doc):
    fields_dict = {"doc": sample_doc, "query": "Good to go?"}
    result = create_prompt_for_travel_advice_response(fields_dict)
    assert result is not None
