from typing import List

from pydantic import BaseModel, Field

from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_core.tools import tool, StructuredTool
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables.base import RunnableBinding, RunnableSequence
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents.base import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_transformers import BeautifulSoupTransformer 

def get_function_calling_model(
    chat_model: BaseChatModel,
    calling_functions: List[StructuredTool],
) -> RunnableBinding:
    if not hasattr(chat_model, "bind_functions"):
        raise ValueError("The provide chat model does not support binding functions. ")
    return chat_model.bind_functions(functions=calling_functions)


class AdviceInput(BaseModel):
    region: str = Field(
        description="""
    Lower case continent name that the search country resides in.
    it should be in one of the following 6 regions:
    africa, americas, asia, europe, middle-east, pacific
    """
    )
    country: str = Field(
        description="""
    Lower case country full name that needs advice of trevel.
    if the country name have multiple words, connect the space with `-`.
    
    Examples: 
    1. usa -> united-states-america
    2. People's Republic of China -> china
    """
    )


@tool(args_schema=AdviceInput)
def advice_whether_safe_to_travel(region: str, country: str) -> str:
    """Check if it is safe to travel to the country of the region"""
    return "Exercise normal safety precautions"


def get_function_calling_model(
    chat_model: BaseChatModel,
    calling_functions: List[StructuredTool],
) -> RunnableBinding:
    if not hasattr(chat_model, "bind_functions"):
        raise ValueError("The provide chat model does not support binding functions. ")
    return chat_model.bind_functions(functions=calling_functions)


def construct_chain_for_getting_advice_url(chatmodel:RunnableBinding) -> RunnableSequence:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a trip advisor which informs whether it is safe to travel to certain country"),
        ("user", "{query}"),
    ])
    
    chain = prompt | chatmodel | OpenAIFunctionsAgentOutputParser()
    return chain


def get_url_for_travel_advice(region: str, country: str) -> str:
    """Get the url for travel advice"""
    return f"https://www.smartraveller.gov.au/destinations/{region}/{country}"


def load_from_url(url: str) -> Document:
    """Get the advice for travel"""
    try:
        html_content = WebBaseLoader(url).load()
    except Exception as e:
        raise Exception(f"Error retrieving web content: {str(e)}")
    return html_content

def transform_html_content(html_content: Document) -> Document:
    transformer = BeautifulSoupTransformer()
    docs_transformed = transformer.transform_documents(
        html_content, tags_to_extract=["p", "li", "div", "a"]
    )
    return docs_transformed[0]

def get_smart_traveler_doc_for_advice(
    user_query: str,
    chat_model: RunnableBinding,
    calling_functions: List[StructuredTool],
    ) -> str:
    """Get the advice for travel"""
    
    chat_model_with_tool = get_function_calling_model(
        chat_model=chat_model, 
        calling_functions=calling_functions)
    chain = construct_chain_for_getting_advice_url(chat_model_with_tool)
    result = chain.invoke({"query": user_query})
    url = get_url_for_travel_advice(**result.tool_input)
    html_content = load_from_url(url)
    smart_traverl_doc_for_advice = transform_html_content(html_content)
    return smart_traverl_doc_for_advice