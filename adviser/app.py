from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

from adviser.advise_model import construct_query2advice_chain
from adviser.utils import detect_injection


def make_app():
    # initialize the llm chat model
    chat_model = ChatOpenAI(temperature=0, model="gpt-4o-mini-2024-07-18")

    class AppQuery(BaseModel):
        query: str

    app = FastAPI()

    @app.get("/health_check")
    async def get_health():
        """health check endpoint"""
        return "ok"

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
            return {"response": "Plase don't try to inject commands."}
        try:
            query2advice_chain = construct_query2advice_chain(chat_model)
            response = query2advice_chain.invoke({"query": user_query})
            return {"response": response}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


app = make_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
