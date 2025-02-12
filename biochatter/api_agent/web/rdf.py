from biochatter.api_agent.base.agent_abc import (
    BaseFetcher,
    BaseInterpreter,
    BaseQueryBuilder,
)
from collections.abc import Callable
import uuid

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.openai_functions import create_structured_output_runnable
from biochatter.llm_connect import Conversation
import requests

RDF_QUERY_PROMPT = """
You are the world class RDF interpreter, and would never make up information!
You have to extract the appropriate information out of the provided context to answer the question.
If applicable, return labels and not codes or ids.

an example query could be:

Wikidata is a graph database where entities (items) are represented by Q-codes (e.g., Q42 for Douglas Adams), and relationships are defined by P-codes (e.g., P31 for "instance of"). Queries use triples in the format:

?subject ?predicate ?object
For example:

?item wdt:P31 wd:Q146.  # Find all items that are "cats"
Key Query Patterns
Find an entity by label (without knowing the Q-ID):

SELECT ?item ?itemLabel WHERE {
  ?item rdfs:label "cat"@en.
}
This retrieves the Q-ID of anything labeled "cat" in English.

Find all instances of a concept (without Q-ID):


SELECT ?item ?itemLabel WHERE {
  ?concept rdfs:label "cat"@en.
  ?item wdt:P31 ?concept.
}
This finds all entities classified as "cats."

Retrieve all known properties of an entity:

SELECT ?property ?propertyLabel ?value ?valueLabel WHERE {
  wd:Q42 ?property ?value.
}
This lists everything known about Douglas Adams (Q42).

Search for labels with a partial match (case-insensitive):

SELECT ?item ?itemLabel WHERE {
  ?item rdfs:label ?itemLabel.
  FILTER(CONTAINS(LCASE(?itemLabel), "cat"))
}
This finds all items whose label contains "cat" (e.g., "Cathedral").

Find people who are scientists:

SELECT ?person ?personLabel WHERE {
  ?person wdt:P31 wd:Q5.  # Must be a human
  ?person wdt:P106 wd:Q901.  # Occupation = scientist
}
This retrieves all humans (Q5) who have the occupation scientist (Q901).

General Query Template:
For broad searches:

SELECT ?subject ?subjectLabel ?predicate ?object ?objectLabel WHERE {
  ?subject ?predicate ?object.
}
LIMIT 100
This lists 100 random relationships in Wikidata.



Example:
#Humans born in New York City
#title: Humans born in New York City
SELECT DISTINCT ?item ?itemLabel ?itemDescription ?sitelinks
WHERE {
    ?item wdt:P31 wd:Q5;            # Any instance of a human
          wdt:P19/wdt:P131* wd:Q60; # Who was born in any value (eg. a hospital)
# that has the property of 'administrative area of' New York City or New York City itself.

# Note that using wdt:P19 wd:Q60;  # Who was born in New York City.
# Doesn't include humans with the birth place listed as a hospital
# or an administrative area or other location of New York City.

          wikibase:sitelinks ?sitelinks.

    SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],mul,en" }
}
ORDER BY DESC(?sitelinks)

EXAMPLE:
#Largest cities of the world
#defaultView:BubbleChart
SELECT ?cityLabel ?population ?gps
WITH {
  SELECT DISTINCT *
  WHERE {
    ?city wdt:P31/wdt:P279* wd:Q515 .
    ?city wdt:P1082 ?population .
    ?city wdt:P625 ?gps .
  }
  ORDER BY DESC(?population)
  LIMIT 100
} AS %i
WHERE {
  INCLUDE %i
  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],mul,en" . }
}
ORDER BY DESC(?population)

"""

RDF_SUMMARY_PROMPT = """
You have to answer this question in a clear and consise manner: {question}. Be factual!
You are the world class RDF interpreter, and would never make up information! You only use 
the provided context to answer the question. Here is the context you have to work with:
{context} 
"""


class RdfQueryParameters(BaseModel):
    base_url: str = Field(
        default="https://query.wikidata.org/sparql",
        description="The base URL of the wikidata SPARQL endpoint",
    )
    endpoint: str = Field(..., description="The name of the SPARQL endpoint")
    query: str = Field(..., description="The SPARQL query to be executed")
    label: str = Field(..., description="The label of an entity")
    limit: int = Field(..., description="The number of results to return")
    language: str = Field(..., description="The language of the entity label")
    question_uuid: str | None = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the question.",
    )

class RdfQueryBuilder(BaseQueryBuilder):
    def create_runnable(
        self, query_parameters: "RdfQueryParameters", conversation: "Conversation"
    ):
        print("Creating runnable")
        print(query_parameters)
        print("prompt", self.structured_output_prompt)
        return create_structured_output_runnable(
            output_schema=query_parameters,
            llm=conversation.chat,
            prompt=self.structured_output_prompt,
        )
    
    def parameterise_query(self, question, conversation):
        runnable = self.create_runnable(
            query_parameters=RdfQueryParameters,
            conversation=conversation,
        )
        rdf_call_obj = runnable.invoke(
            {"input": f"Answer:\n{question} based on:\n{RDF_QUERY_PROMPT}"}
        )
        rdf_call_obj.question_uuid = str(uuid.uuid4())
        return [rdf_call_obj]

class RdfFetcher(BaseFetcher):
    """A fetcher for RDF data"""

    def __init__(self, api_token="demo"):
        self.headers = {
            "Accept": "application/sparql-results+json"
        }
        self.base_url = "https://query.wikidata.org/sparql"

    def fetch_results(
        self,
        request_data: RdfQueryParameters,
        retries: int | None = 3,
    ) -> str:
        """Fetch RDF data from the specified endpoint.
        Args:
        ----
        request_data: RdfQueryParameters
            The query parameters
        Returns:
        -------
            str: The RDF data
        """
        query = request_data[0]
        # submit the query and get the URL
        params = query.dict(exclude_unset=True)
        full_url = f"{self.base_url}"
        response = requests.get(full_url, headers=self.headers, params=params)
        response.raise_for_status()
        # return response
        # Fetch the results from the URL
        results_response = requests.get(response.url, headers=self.headers)
        return results_response.json()


class RdfInterpreter(BaseInterpreter):
    def summarise_results(
        self, question: str, conversation_factory: Callable, response_text: str
    ) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a world class RDF interpreter! "
                    "Your task is to interpret the results from "
                    "the API calls and to summarise them for the user.",
                ),
                ("user", "{input}"),
            ]
        )
        summary_prompt = RDF_SUMMARY_PROMPT.format(
            question=question,
            context=response_text,
        )
        output_parser = StrOutputParser()
        conversation = conversation_factory()
        chain = prompt | conversation.chat | output_parser
        answer = chain.invoke({"input": {summary_prompt}})
        return answer
