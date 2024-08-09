from unittest.mock import patch, Mock, MagicMock
from typing import Dict

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI

from adviser.app import make_app
from adviser.config import INJECTION_PATTERNS


# Create the FastAPI app instance
@pytest.fixture
@patch("adviser.app.construct_query2advice_chain")
def app(mock_chain_constructor: Mock):
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
    assert response.json() == {'status': 'ok'} 


@patch("adviser.app.construct_query2advice_chain")
def test_get_travel_advice_endpoint_success(
    mock_query2advice_chain_constructor: Mock, client: TestClient, query_data: Dict
):
    mock_query2advice_chain_constructor.return_value.invoke.return_value = "Good to go"
    response = client.post("/get_travel_advice", json=query_data)
    assert response.status_code == 200
    assert response.json() == {"response": "Good to go"}


def test_get_travel_advice_endpoint_invalid_input(client: TestClient):
    # test no input
    response = client.post("/get_travel_advice", json={"query": ""})
    assert response.status_code == 400
    assert response.json() == {"detail": "Query is required"}


@pytest.mark.parametrize(
    "query",
    INJECTION_PATTERNS,
)
def test_get_travel_advice_endpoint_prompt_injection(query: str, client: TestClient):
    # test injection input
    response = client.post(
        "/get_travel_advice", json={"query": query + " travel advice"}
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Injection commands detected."}


@patch("adviser.app.construct_query2advice_chain")
def test_handle_query_endpoint_exception(
    mock_query2advice_chain_constructor: Mock, client: TestClient, query_data: Dict
):
    mock_query2advice_chain_constructor.return_value.invoke.side_effect = Exception(
        "Some error"
    )
    response = client.post("/get_travel_advice", json=query_data)
    assert response.status_code == 500
    assert response.json() == {"detail": "Some error"}
