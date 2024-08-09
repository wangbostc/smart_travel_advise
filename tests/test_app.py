from unittest.mock import patch, Mock
from typing import Dict

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI

from adviser.app import make_app
from adviser.config import INJECTION_PATTERNS


# Create the FastAPI app instance
@pytest.fixture
def app():
    return make_app()


# Create a TestClient instance
@pytest.fixture
def client(app: FastAPI):
    return TestClient(app)


@pytest.fixture
def query_data():
    return {"query": "travel advice"}


def test_health_check_endpoint(client: TestClient):
    response = client.get("/health_check")
    assert response.status_code == 200
    assert response.json() == "ok"


def test_set_api_key(client: TestClient):
    # Test valid API key
    response = client.post("/set_api_key", json={"key": "test-api-key"})
    assert response.status_code == 200
    assert response.json() == {"message": "API key sets up successfully"}

    # Test missing API key
    response = client.post("/set_api_key", json={"key": ""})
    assert response.status_code == 400
    assert response.json() == {"detail": "API key is required"}


@patch("adviser.app.construct_query2advice_chain")
def test_get_travel_advice_endpoint_success(
    mock_query2advice_chain_constructor: Mock, client: TestClient, query_data: Dict
):
    client.post("/set_api_key", json={"key": "test-api-key"})
    mock_query2advice_chain_constructor.return_value.invoke.return_value = "Good to go"
    response = client.post("/get_travel_advice", json=query_data)
    assert response.status_code == 200
    assert response.json() == {"response": "Good to go"}


@patch("adviser.app.construct_query2advice_chain")
def test_get_travel_advice_endpoint_invalid_input(
    mock_query2advice_chain_constructor: Mock, client: TestClient
):
    client.post("/set_api_key", json={"key": "test-api-key"})
    mock_query2advice_chain_constructor.return_value.invoke.return_value = "Good to go"
    # test no input
    response = client.post("/get_travel_advice", json={"query": ""})
    assert response.status_code == 400
    assert response.json() == {"detail": "Query is required"}


@pytest.mark.parametrize(
    "query",
    INJECTION_PATTERNS,
)
def test_get_travel_advice_endpoint_prompt_injection(query: str, client: TestClient):
    client.post("/set_api_key", json={"key": "test-api-key"})
    # test injection input
    response = client.post(
        "/get_travel_advice", json={"query": query + " travel advice"}
    )
    assert response.status_code == 200
    assert response.json() == {"response": "Plase don't try to inject commands."}


@patch("adviser.app.construct_query2advice_chain")
def test_handle_query_endpoint_exception(
    mock_query2advice_chain_constructor: Mock, client: TestClient, query_data: Dict
):
    client.post("/set_api_key", json={"key": "test-api-key"})
    mock_query2advice_chain_constructor.return_value.invoke.side_effect = Exception(
        "Some error"
    )

    response = client.post("/get_travel_advice", json=query_data)
    assert response.status_code == 500
    assert response.json() == {"detail": "Some error"}
