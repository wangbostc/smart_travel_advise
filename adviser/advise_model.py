from typing import Dict, Optional, Union

from pydantic import BaseModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.documents.base import Document
from langchain_core.runnables import (
    RunnableSequence,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

from adviser.adviser_support_info_retriver import (
    construct_query2url_chain,
    construct_url2doc_chain,
)
from adviser.utils import extract_content_from_text


class TravelAdviceInput(BaseModel):
    """
    Represents the input data for generating travel advice prompt.

    Args:
        name (str): The name of the travel advice.
        html_section (str): The HTML section of the travel advice. (for query it will just return the query)
        html_section_start (Optional[str], optional): The start of the HTML section. Defaults to None.
        html_section_end (Optional[str], optional): The end of the HTML section. Defaults to None.
        html_section_length (Optional[int], optional): The length of the HTML section. Defaults to None.
    """

    name: str
    html_section: str
    html_section_start: Optional[str] = None
    html_section_end: Optional[str] = None
    html_section_length: Optional[int] = None


def define_information_input_variables() -> Dict[str, TravelAdviceInput]:
    """Defines the input to prompt and how they should be extracted from the text document."""
    input_variables = {
        "title": TravelAdviceInput(name="title", html_section="metadata"),
        "description": TravelAdviceInput(name="description", html_section="metadata"),
        "latest_update": TravelAdviceInput(
            name="latest_update",
            html_section="page_content",
            html_section_start="Latest update",
            html_section_end="Download",
            html_section_length=800,
        ),
        "advice_levels": TravelAdviceInput(
            name="advice_levels",
            html_section="page_content",
            html_section_start="Advice levels",
            html_section_end="Overview",
            html_section_length=500,
        ),
        "query": TravelAdviceInput(name="query", html_section="query"),
    }

    return input_variables


def get_required_prompt_field(
    input_variable: TravelAdviceInput, doc: Document, query: str
) -> str:
    """The function to get the required prompt field from the document and user query.

    Args:
        input_variable_retrieve_method (List[Union[str, int]]): the method to retrieve the input variable.
        doc (Document): the document to extract the information from.
        query (str): the original user query.
    Raises:
        ValueError: if the keywords is not in current retrieval ["query", "metadata", "page_content"].
    """
    match input_variable.html_section:
        case "query":
            return query
        case "metadata":
            return doc.metadata.get(input_variable.name, "")
        case "page_content":
            return extract_content_from_text(
                text=doc.page_content,
                start_word=input_variable.html_section_start,
                end_word=input_variable.html_section_end,
                extraction_length=input_variable.html_section_length,
            )
        case _:
            raise ValueError("The input variable retrieval is not supported")


def get_required_prompt_fields(doc: Document, query: str) -> Dict[str, str]:
    """Retrieves the required prompt fields based on the given document and query."""
    input_variables = define_information_input_variables()
    return {
        key: get_required_prompt_field(value, doc, query)
        for key, value in input_variables.items()
    }


def create_prompt_template_for_travel_advice() -> PromptTemplate:
    """Creates a prompt template for generating travel advice."""
    template = """ 
        Given the following inputs:
        
        `title`: The title of the webpage.
        `description`: A brief description of the webpage content.
        `latest_update`: The latest update of the travel advisory for the country.
        `advice_levels`: The score level of the country and its sub-regions.
        `query`: The question that the traveller asked about the country he is going to travel to.
        
        ===================================================================================================
        
        Your task is to assess whether it is safe to travel to the specified country and provide reasons for your assessment
        that is related to the `latest_update`.
        If a sub-region is specified, also provide the advice level for that sub-region. 
        Do not list advice levels for sub-regions that are not specified or the information is not provided.
        
        Your responsed advice should be one of following:
        ['Do Not Travel', 'Reconsider Your Need to Travel', 'Exercise a High Degree of Caution', 'Exercise Normal Safety Precautions']
        If the `title` returns "page not found", respond with: "There is no trip advisory based on the current information."
        
        ===================================================================================================

        Example:
        
        Give following inputs:

        `title`: "Indonesia Travel Advice & Safety | Smartraveller"
        `description`: "Australian Government travel advice for Indonesia. Exercise a high degree of caution. Travel advice level YELLOW. 
        Understand the risks, safety, laws and contacts."
        `latest_update`: "The Bali Provincial Government has introduced a new tourist levy of IDR 150,000 per person to foreign tourists entering Bali. 
        The tourist levy is separate from the e-Visa on Arrival or the Visa on Arrival. Cashless payments can be made online prior to travel or on arrival 
        at designated payment counters at Bali's airport and seaport. See the Bali Provincial Government's official website for further information (see 
        link in 'Travel' section below)."
        `advice_levels`: "Exercise a high degree of caution" in Indonesia overall.\n"Reconsider Your Need to Travel" in Papua.
        
        with different queries, you should give different responses:
        Case 1:
        `query`: "I would like to travel to Indonesia. Is it safe?"
        `response`:
        Travel Safety Level: 
            "Exercise a high degree of caution" in Indonesia overall.
        Reasons:
            ongoing risk of terrorist attack.
            the risk of serious security incidents or demonstrations that may turn violent.
        Case 2:
        `query`: "I would like to travel to Papua in Indonesia. Is it safe?"
        `response`:
        Travel Safety Level: 
            "Exercise a high degree of caution" in Indonesia overall.
            "Reconsider Your Need to Travel" in Papua.
        Reasons:
            ongoing risk of terrorist attack.
        Case 3:
        `query`: "I would like to travel to Jakarta in Indonesia. Is it safe?"
        `response`:
       Travel Safety Level: 
            "Exercise a high degree of caution" in Indonesia overall.
        Reasons:
            ongoing risk of terrorist attack.
        Since no specific advice level for Jakarta is provided, the overall advice for Indonesia applies.

        ===================================================================================================
        
        Please complete the task using following information:
        `title`: {title}
        `description`:{description}
        `latest_update`: {latest_update}
        `advice_levels`: {advice_levels}
        `query`: {query}
        
        Please generate the response, if information is not sufficient to complete the task do not make up 
        the answer, and response with "I do not have sufficient information to provide you with the advice."
        If {query} is not related to travel or travel advice, respones with "Please only ask travel advice related questions."
        """

    prompt_template = PromptTemplate(
        input_variables=define_information_input_variables().keys,
        template=template,
    )

    return prompt_template


def create_prompt_for_travel_advice_response(
    fields_dict: Dict[str, Union[Document, Dict[str, str]]]
) -> RunnableSequence:
    """Creates a prompt for generating a travel advice response."""
    required_fields = get_required_prompt_fields(**fields_dict)

    prompt_template = create_prompt_template_for_travel_advice()
    prompt = prompt_template.format(**required_fields)
    return prompt


def construct_doc2advice_chain(chat_model: BaseChatModel):
    """
    Construct the chain that takes in a dictionary of doc (support information)
    and the original query provided by the user. The chain will output the final advice.
    """
    doc2advice_chain = (
        RunnableLambda(create_prompt_for_travel_advice_response)
        | chat_model
        | StrOutputParser()
    )

    return doc2advice_chain


def construct_query2advice_chain(chat_model: BaseChatModel):
    """
    Constructs a end to end query to advice chain for the given chat model.
    Parameters:
        chat_model (BaseChatModel): The chat model to be used for constructing the chain.
        this model should not have tool binding to avoid unnecessary tool calling.

    Returns:
        RunnableParallel: The query to advice chain.
    """

    query2url_chain = construct_query2url_chain(chat_model)
    url2doc_chain = construct_url2doc_chain()
    doc2advice_chain = construct_doc2advice_chain(chat_model)
    query2advice_chain = (
        RunnableParallel(
            {
                "doc": query2url_chain | url2doc_chain,
                "query": RunnablePassthrough(),
            }
        )
        | doc2advice_chain
    )
    return query2advice_chain
