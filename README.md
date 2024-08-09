# Smart Travel Advise

Smart Travel Advise is a FastAPI-based application that provides travel advice based on user queries. It leverages machine learning models to generate personalized travel recommendations.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Testing](#testing)

## Installation

### Prerequisites

- Python 3.11
- pip (Python package installer)

### Steps

1. Clone the repository:

    ```sh
    git clone https://github.com/wangbostc/smart_travel_advise.git
    cd smart_travel_advise
    ```

2. Create a virtual environment:

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the dependencies:

    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Start the FastAPI server:

    ```sh
    uvicorn adviser.app:app --host 0.0.0.0 --port 8000
    ```

2. Open your browser and navigate to `http://0.0.0.0:8000/docs` to access the interactive API documentation.

## API Endpoints

### `GET /health_check`

- **Description**: For a quick health check.
- **Response**: JSON object indicating the health status.

#### Example

**Response**:

```json
{
    "status": "ok"
}
```

### `POST /get_travel_advice`

- **Description**: Provides travel advice based on the user's query.
- **Request Body**: JSON object containing the query.
- **Response**: JSON object with travel advice.

#### Example

**Request**:

```json
{
    "query": "I would like to travel to Indonesia. Is it safe?"
}
```

**Response**:

```json
{
    "response": "Travel Safety Level: \n\"Exercise a high degree of caution\" in Indonesia overall.\n\nReasons:\n- Ongoing security risks, including the potential for terrorist attacks.\n- Higher levels of caution are advised in certain areas due to the risk of serious security incidents or demonstrations that may turn violent."
}
```

## Testing

To run the tests, use the following command:
    ```sh
    pytest
    ```