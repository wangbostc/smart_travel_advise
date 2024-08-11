from unittest.mock import patch, Mock, MagicMock
from typing import Dict

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI

from adviser.config import INJECTION_PATTERNS
from adviser.make_app import make_app


@pytest.fixture
def mock_chain() -> MagicMock:
    return MagicMock()


# Create the FastAPI app instance
@pytest.fixture
def app(mock_chain: Mock):
    return make_app(mock_chain)


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
    assert response.json() == {"status": "ok"}


def test_get_travel_advice_endpoint_success(
    mock_chain: Mock, client: TestClient, query_data: Dict
):
    mock_chain.invoke.return_value = "Good to go"
    response = client.post("/get_travel_advice", json=query_data)
    assert response.status_code == 200
    assert response.json() == {"response": "Good to go"}


def test_get_travel_advice_endpoint_invalid_input(client: TestClient):
    # test no input
    response = client.post("/get_travel_advice", json={"query": ""})
    assert response.status_code == 400
    assert response.json() == {"detail": "Query is required."}


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


def test_handle_query_endpoint_exception(
    mock_chain: Mock, client: TestClient, query_data: Dict
):
    mock_chain.invoke.side_effect = Exception("Some error")
    response = client.post("/get_travel_advice", json=query_data)
    assert response.status_code == 500
    assert response.json() == {"detail": "Some error"}
