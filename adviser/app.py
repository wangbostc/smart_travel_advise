from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from adviser.config import OPENAI_API_KEY
from adviser.smart_traveler_info_retriver import (
    advice_whether_safe_to_travel,
    construct_chain_for_advice_doc_from_url,
    construct_chain_for_getting_advice_url,
)
from adviser.advise_model import (
    create_prompt_for_travel_advice_response,
)


def make_app():
    # initialize the llm chat model
    chat_model = ChatOpenAI(temperature=0, model="gpt-4o-mini-2024-07-18")

    class AppQuery(BaseModel):
        query: str

    app = FastAPI()

    @app.get("/health_check")
    async def get_health():
        return "ok"

    @app.post("/get_advice")
    async def get_advice(query: AppQuery):

        chain_query2url = construct_chain_for_getting_advice_url(
            chat_model=chat_model, calling_tool=advice_whether_safe_to_travel
        )
        chain_url2doc = construct_chain_for_advice_doc_from_url()
        full_chain = (
            RunnableParallel(
                {
                    "doc": chain_query2url | chain_url2doc,
                    "query": RunnablePassthrough(),
                }
            )
            | RunnableLambda(create_prompt_for_travel_advice_response)
            | chat_model
            | StrOutputParser()
        )
        
        query = query.query
        response = full_chain.invoke({'query': query})
            
        return {"response": response}
    
    return app

app = make_app()
