from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

from adviser.advise_model import construct_query2advice_chain
from adviser.utils import detect_injection


class AppQuery(BaseModel):
    query: str

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "query": "I would like to travel to Indonesia. Is it safe?",
                }
            ]
        }


def make_app():
    # initalize chain
    chat_model = ChatOpenAI(temperature=0, model="gpt-4o-mini-2024-07-18")

    app = FastAPI(
        title="Travel Advice API",
        description="""
        This API is for retrieving travel advice based on user queries.
        It leverages the LangChain library and LLM from OPENAI to generate responses based on the user's query.
        The advice given is based on the information provided from https://www.smartraveller.gov.au/
        """,
        version="0.0.1",
        docs_url="/",
        openapi_tags=[
            dict(
                name="Get Trip Advice Endpoint",
                externalDocs=dict(
                    description="Website for travel advice augmentment",
                    url="https://www.smartraveller.gov.au/destinations",
                ),
            ),
        ],
    )

    @app.get(
        "/health_check",
        summary="Get the health status of the API",
        tags=["Health Check Endpoint"],
        responses={
            200: {
                "description": "Successful Response",
                "content": {"application/json": {"example": {"status": "ok"}}},
            },
        },
    )
    async def get_health():
        """health check endpoint"""
        return {"status": "ok"}

    @app.post(
        "/get_travel_advice",
        summary="Endpiont provides LLM powered travel advice",
        description="Retrieve travel advice based on the provided user query.",
        response_description="The travel advice given by LLM.",
        tags=["Get Trip Advice Endpoint"],
        responses={
            200: {
                "description": "Successful Response",
                "content": {
                    "application/json": {
                        "example": {
                            "response": 'Travel Safety Level:\n"Exercise a high degree of caution" in Indonesia overall.\nReasons: \nongoing risk of terrorist attack.'
                        }
                    }
                },
            },
            400: {
                "description": "Bad Request",
                "content": {
                    "application/json": {
                        "example": [
                            {"detail": "Query is required."},
                            {"detail": "Injection commands detected."},
                        ]
                    }
                },
            },
            500: {
                "description": "Internal Server Error",
                "content": {
                    "application/json": {
                        "example": {
                            "detail": "An error occurred while processing your request."
                        }
                    }
                },
            },
        },
    )
    async def get_travel_advice(query: AppQuery):
        """Retrieves travel advice based on the provided query."""
        user_query = query.query
        if not user_query:
            raise HTTPException(status_code=400, detail="Query is required.")
        if detect_injection(user_query):
            raise HTTPException(status_code=400, detail="Injection commands detected.")
        try:
            query2advice_chain = construct_query2advice_chain(chat_model)
            response = query2advice_chain.invoke({"query": user_query})
            return {"response": response}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


app = make_app()
