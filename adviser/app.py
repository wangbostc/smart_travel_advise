from adviser.make_app import make_app
from langchain_openai import ChatOpenAI
from adviser.advise_model import construct_query2advice_chain

chat_model = ChatOpenAI(temperature=0, model="gpt-4o-mini-2024-07-18")
query2advice_chain = construct_query2advice_chain(chat_model)
app = make_app(chain=query2advice_chain)
