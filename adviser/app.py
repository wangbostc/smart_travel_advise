from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

from adviser.config import OPENAI_API_KEY
from adviser.advise_model import construct_query2advice_chain


def make_app():
    # initialize the llm chat model
    chat_model = ChatOpenAI(temperature=0, model="gpt-4o-mini-2024-07-18")

    class AppQuery(BaseModel):
        query: str

    app = FastAPI()

    @app.get("/health_check")
    async def get_health():
        return "ok"

    @app.post("/get_travel_advice")
    async def get_travel_advice(query: AppQuery):
        query2advice_chain = construct_query2advice_chain(chat_model)
        user_query = query.query
        if not user_query:
            raise HTTPException(status_code=400, detail="Query is required")
        try:
            response = query2advice_chain.invoke({"query": user_query})
            return {"response": response}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


app = make_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
