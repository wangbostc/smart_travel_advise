from typing import List, Dict

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_core.documents.base import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class Country(BaseModel):
    name: str = Field(
        description="""
    Lower case country full name that needs advice of trevel.
    if the country name have multiple words, connect the space with `-`.
    
    Examples: 
    1. usa -> united-states-america
    2. People's Republic of China -> china
    """
    )
    region: str = Field(
        description="""
    Lower case continent name that the search country resides in.
    it should be in one of the following 6 regions:
    [africa, americas, asia, europe, middle-east, pacific]
    """
    )


def get_url_for_travel_advice(country: Country) -> str:
    """Get the url for the travel advice, receives the location dictionary and returns the url"""
    region = country.region
    country = country.name
    if (not region) | (not country):
        raise ValueError("Please provide the region and country for travel advice")
    return f"https://www.smartraveller.gov.au/destinations/{region}/{country}"


def construct_query2url_chain(chat_model: BaseChatModel) -> RunnableSequence:
    """Construct the chain for retrieving the URL for travel advice from a query."""
    parser = PydanticOutputParser(pydantic_object=Country)
    prompt = PromptTemplate(
        template="""Find the country and its region that the user is asking for travel advice.
        Be careful, Indonesia belongs to the asia region, not pacific.
        {format_instructions}\n{query}""",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    query2url = prompt | chat_model | parser | RunnableLambda(get_url_for_travel_advice)
    return query2url


def load_from_url(url: str) -> Document:
    """Retrieve the advice HTML content for travel from the given URL."""
    try:
        html_content = WebBaseLoader(url).load()
    except Exception as e:
        raise Exception(f"Error retrieving web content: {str(e)}")
    return html_content


def transform_html_content(html_content: Document) -> Document:
    """Transform the HTML content to text content."""
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(html_content)
    return docs_transformed[0]


def construct_url2doc_chain() -> Document:
    """Construct the chain for retrieving the advice text document from a URL."""
    url2doc_chain = RunnableLambda(load_from_url) | RunnableLambda(
        transform_html_content
    )
    return url2doc_chain
