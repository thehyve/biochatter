import os
from collections.abc import Callable
from rdflib import Graph, RDF, RDFS, OWL

from .llm_connect import Conversation, GptConversation


class SparqlPromptEngine:
    def __init__(self, 
                 ttl_file_path: str = None,
                 model_name: str = "gpt-3.5-turbo",
                 conversation_factory: Callable = None,
                 ):
        
        if not ttl_file_path:
            raise ValueError("ttl_file_path is required.")


        self.conversation_factory = conversation_factory if conversation_factory is not None else self._get_conversation
    
        schema = Graph()
        schema.parse(ttl_file_path, format="turtle")

        self.entities = set(schema.subjects(RDF.type, OWL.Class)).union(set(schema.subjects(RDF.type, RDFS.Class)))
        self.entities = [str(entity) for entity in self.entities]
        self.relationships = {str(cl): [str(l) for l in list(schema.predicates(subject=cl))] for cl in self.entities}
        self.selected_entities = []
        self.selected_relationships = []
        self.question = ""
        self.model_name = model_name

    def _get_conversation(
        self,
        model_name: str | None = None,
    ) -> "Conversation":
        """Create a conversation object given a model name.

        Args:
        ----
            model_name: The name of the model to use for the conversation.

        Returns:
        -------
            A BioChatter Conversation object for connecting to the LLM.

        Todo:
        ----
            Genericise to models outside of OpenAI.

        """
        conversation = GptConversation(
            model_name=model_name or self.model_name,
            prompts={},
            correct=False,
        )
        conversation.set_api_key(
            api_key=os.getenv("OPENAI_API_KEY"),
            user="test_user",
        )
        return conversation


    def _select_entities(
            self,
            question: str,
            conversation: "Conversation",
    ) -> bool:
        self.question = question
        
        conversation.append_system_message(
            "You have access to a RDF graph that contains "
            f"these classes: {', '.join(self.entities)}. Your task is "
            "to select the classes that are relevant to the user's question "
            "for subsequent use in a query. Only return the full IRI of the classes, "
            "comma-separated, without any additional text. Do not return "
            "names, relationships, or properties.",
        )

        msg, token_usage, correction = conversation.query(question)

        result = msg.split(",") if msg else []
        # TODO: do we go back and retry if no entities were selected? or ask for
        # a reason? offer visual selection of entities and relationships by the
        # user?

        print("result", result)
        print("entities", self.entities)

        if result:
            for entity in result:
                entity = entity.strip()
                if entity in self.entities:
                    self.selected_entities.append(entity)
        return bool(result)
    
    
    def _select_relationships(self, conversation: "Conversation") -> bool:
        """Given a question and the preselected entities, select relationships for
        the query.

        Args:
        ----
            conversation: A BioChatter Conversation object for connecting to the
                LLM.

        Returns:
        -------
            True if at least one relationship was selected, False otherwise.

        Todo:
        ----
            Now we have the problem that we discard all relationships that do
            not have a source and target, if at least one relationship has a
            source and target. At least communicate this all-or-nothing
            behaviour to the user.

        """
        if not self.question:
            raise ValueError(
                "No question found. Please make sure to run entity selection first.",
            )

        if not self.selected_entities:
            raise ValueError(
                "No entities found. Please run the entity selection step first.",
            )
        
        print("selected entities", self.selected_entities)
        rels = {}

        for entity in self.selected_entities:
            rels[entity] = self.relationships[entity]

        msg = (
            "You have access to a knowledge graph that contains "
            f"these entities: {', '.join(self.selected_entities)}. "
            "Your task is to select the relationships that are relevant "
            "to the user's question for subsequent use in a query. Only "
            "return the relationships without their sources or targets, "
            "comma-separated, and without any additional text. Here are the "
            "possible relationships and their source and target entities: "
            f"{rels}."
        )

        conversation.append_system_message(msg)

        res, token_usage, correction = conversation.query(self.question)

        result = res.split(",") if msg else []

        if result:
            for relationship in result:
                relationship = relationship.strip()
                for entity in self.selected_entities:
                    if relationship in self.relationships[entity]:
                        self.selected_relationships.append(relationship)

        return bool(result)
    
    @staticmethod
    def _validate_json_str(json_str: str):
        json_str = json_str.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]
        return json_str.strip()


    def _select_graph_entities_from_question(
        self,
        question: str,
        conversation: Conversation,
    ) -> str:
        conversation.reset()
        success1 = self._select_entities(
            question=question,
            conversation=conversation,
        )
        if not success1:
            raise ValueError(
                "Entity selection failed. Please try again with a different question.",
            )
        conversation.reset()
        success2 = self._select_relationships(conversation=conversation)
        if not success2:
            raise ValueError(
                "Relationship selection failed. Please try again with a different question.",
            )

    def _generate_query_prompt(
            self,
            entities: list,
            relationships: dict,
            query_language: str | None = "SPARQL",
        ) -> str:
            """Generate a prompt for a large language model to generate a database
            query based on the selected entities, relationships, and properties.

            Args:
            ----
                entities: A list of entities that are relevant to the question.

                relationships: A list of relationships that are relevant to the
                    question.

                properties: A dictionary of properties that are relevant to the
                    question.

                query_language: The language of the query to generate.

            Returns:
            -------
                A prompt for a large language model to generate a database query.

            """
            msg = (
                f"Generate a database query in {query_language} that answers "
                f"the user's question. "
                f"You can use the following entities: {entities} and  "
                f"relationships: {relationships}. "
                "A query has to start with defining PREFIXES for the selected entities and relationships."
            )

            msg += "Only return the query, without any additional text, symbols or characters --- just the query statement."
            return msg
    
    def generate_query_prompt(
        self,
        question: str,
        query_language: str | None = "SPARQL",
    ) -> str:
        """Generate a prompt for a large language model to generate a database
        query based on the user's question and class attributes informing about
        the schema.

        Args:
        ----
            question: A user's question.

            query_language: The language of the query to generate.

        Returns:
        -------
            A prompt for a large language model to generate a database query.

        """
        self._select_graph_entities_from_question(
            question,
            self.conversation_factory(),
        )
        msg = self._generate_query_prompt(
            self.selected_entities,
            self.selected_relationships,
            query_language,
        )
        return msg

    def _generate_query(
        self,
        question: str,
        entities: list,
        relationships: dict,
        query_language: str,
        conversation: "Conversation",
    ) -> str:
        """Generate a query in the specified query language that answers the user's
        question.

        Args:
        ----
            question: A user's question.

            entities: A list of entities that are relevant to the question.

            relationships: A list of relationships that are relevant to the
                question.

            properties: A dictionary of properties that are relevant to the
                question.

            query_language: The language of the query to generate.

            conversation: A BioChatter Conversation object for connecting to the
                LLM.

        Returns:
        -------
            A database query that could answer the user's question.

        """
        msg = self._generate_query_prompt(
            entities,
            relationships,
            query_language,
        )

        conversation.append_system_message(msg)

        out_msg, token_usage, correction = conversation.query(question)

        return out_msg.strip()
    
    
    def generate_query(
        self,
        question: str,
        query_language: str | None = "SPARQL",
    ) -> str:
        """Wrap entity and property selection and query generation; return the
        generated query.

        Args:
        ----
            question: A user's question.

            query_language: The language of the query to generate.

        Returns:
        -------
            A database query that could answer the user's question.

        """
        self._select_graph_entities_from_question(
            question,
            self.conversation_factory(),
        )

        return self._generate_query(
            question=question,
            entities=self.selected_entities,
            relationships=self.selected_relationships,
            query_language=query_language,
            conversation=self.conversation_factory(),
        )
