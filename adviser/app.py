from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

from adviser.advise_model import construct_query2advice_chain
from adviser.utils import detect_injection


class AppQuery(BaseModel):
    query: str


def make_app():
    # initalize chain
    chat_model = ChatOpenAI(temperature=0, model="gpt-4o-mini-2024-07-18")

    app = FastAPI()

    @app.get("/health_check")
    async def get_health():
        """health check endpoint"""
        return  {"status": "ok"}

    @app.post("/get_travel_advice")
    async def get_travel_advice(query: AppQuery):
        """
        Retrieves travel advice based on the provided query.
        Args:
            query (AppQuery): The query object containing the user's query.
        Returns:
            dict: A dictionary containing the response from the advice chain.
        Raises:
            HTTPException: If the query is empty, a HTTPException with status code 400 and detail message "Query is required" is raised.
            HTTPException: If an exception occurs during the invocation of the advice chain, a HTTPException with status code 500 and the exception message is raised.
        """
        user_query = query.query
        if not user_query:
            raise HTTPException(status_code=400, detail="Query is required")
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
