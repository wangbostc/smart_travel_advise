from typing import List, Dict

from pydantic import BaseModel, Field
from langchain_core.tools import tool, StructuredTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableBinding, RunnableSequence, RunnableLambda
from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_core.documents.base import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_transformers import Html2TextTransformer


class AdviceInput(BaseModel):
    region: str = Field(
        description="""
    Lower case continent name that the search country resides in.
    it should be in one of the following 6 regions:
    [africa, americas, asia, europe, middle-east, pacific]
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
    """Check if it is safe to travel to the country of the region
    Sometimes the user may ask about a specific sub-region of the country.
    If a match of a sub-region is found, find the country of that sub-region first
    then find the region that country belongs to."""
    return "Exercise normal safety precautions"


def get_tool_calling_model(
    chat_model: BaseChatModel,
    calling_tools: List[StructuredTool],
) -> RunnableBinding:
    """Bind the calling functions to the chat model, here the model is focused on function calling"""
    if not hasattr(chat_model, "bind_functions"):
        raise ValueError("The provide chat model does not support binding functions. ")
    return chat_model.bind_tools(tools=calling_tools, tool_choice="required")


def get_url_for_travel_advice(loc_dict: Dict[str, str]) -> str:
    """Get the url for travel advice"""
    region = loc_dict.get("region", "").lower()
    country = loc_dict.get("country", "").lower()
    if (not region) | (not country):
        raise ValueError("Please provide the region and country for travel advice")
    return f"https://www.smartraveller.gov.au/destinations/{region}/{country}"


def construct_chain_for_getting_advice_url(
    chat_model: RunnableBinding,
    calling_tool: StructuredTool,
) -> RunnableSequence:
    """Construct the chain for getting the advice url
    Leverage chat model tool calling to retrieve the advice url

    Args:
        chat_model (RunnableBinding): the chat_model to
        calling_tool (StructuredTool): _description_

    Returns:
        RunnableSequence: _description_
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a trip advisor which informs whether it is safe to travel to a certain country.
                You have the knowledge of which country the user is asking and which region of this six regions:
                [africa, americas, asia, europe, middle-east, pacific] the country resides in.
                Be careful, Indonesia belongs to the asia region, not pacific.
                """,
            ),
            ("user", "{query}"),
        ]
    )
    chat_model_with_tool = get_tool_calling_model(
        chat_model=chat_model, calling_tools=[calling_tool]
    )

    output_parser = JsonOutputKeyToolsParser(key_name=calling_tool.name)
    # here jsonoutputkeytoolsparsers will return a list of dict, we only need the first one

    chain = (
        prompt
        | chat_model_with_tool
        | output_parser
        | RunnableLambda(lambda x: x[0])
        | RunnableLambda(get_url_for_travel_advice)
    )

    return chain


def load_from_url(url: str) -> Document:
    """Get the advice for travel"""
    try:
        html_content = WebBaseLoader(url).load()
    except Exception as e:
        raise Exception(f"Error retrieving web content: {str(e)}")
    return html_content


def transform_html_content(html_content: Document) -> Document:
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(html_content)
    return docs_transformed[0]


def construct_chain_for_advice_doc_from_url() -> Document:
    """Get the advice information for model to giving user travel advice"""

    chain = RunnableLambda(load_from_url) | RunnableLambda(transform_html_content)
    return chain
