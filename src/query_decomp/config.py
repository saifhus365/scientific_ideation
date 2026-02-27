from dataclasses import dataclass, field
import os
from dotenv import load_dotenv

__all__ = ['LLMConfig', 'PromptTemplates']

@dataclass
class LLMConfig:
    # Load environment variables from a .env file
    load_dotenv()

    model_name: str = "mistral-medium-latest"  # Changed to a standard Mistral model
    temperature: float = 0.1
    max_tokens: int = 500
    api_key: str = field(default_factory=lambda: os.getenv("MISTRAL_API_KEY"))

    def __post_init__(self):
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not found. Please set it in your environment variables or a .env file.")

@dataclass
class PromptTemplates:
    query_analysis: str = """
        Analyze this research paper query and output ONLY a JSON object with this exact structure:

            "topics": ["topic1", "topic2"],
            "timeline": 
                "start_date": null,
                "end_date": null,
                "specific_year": null
            ,
            "intention": "intention class"



        Rules:
        1. Topics must be 1-3 word phrases, not sentences. I DO NOT want more than 3 topics.
        2. Dates must be in YYYY-MM-DD format
        3. Specific_year must be a number
        4. If the dates are not explicitly specified in the query, I want you to use your knowledge about the topic and assign some dates. If you yourself are not sure, Then you can assign the dates as NULL.
        4. Intention must be exactly one of the following classes: Knowledge Acquisition, Planning, Comparison. I want you to understand which one of the two the user wants to perform:
            - Exploratory: The user seeks to discover patterns, generate hypotheses, or broadly explore a topic without predefined expectations. Often open-ended and inductive.
            - Comparative: The user intends to compare two or more methods, approaches, datasets, tools, methods, or hypotheses to evaluate differences or effectiveness.
            - Descriptive: The user wants a concise, factual explanation or summary of a concept, object, or phenomenon—often 'What is X?'.
            - Causal: The user asks about causes, effects, or mechanisms—typically beginning with 'Why' or 'How does X influence Y?'.
            - Relational: The user inquires about the relationship or correlation between two or more scientific variables or phenomena.
        5. Output ONLY the JSON object, no other text

        Query: {query}
        """