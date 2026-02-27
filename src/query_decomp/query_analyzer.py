from .config import LLMConfig, PromptTemplates
from .llm_client import LLMClient
from .response_parser import ResponseParser, QueryAnalysis

class QueryAnalyzer:
    def __init__(self):
        self.config = LLMConfig()
        self.templates = PromptTemplates()
        self.llm_client = LLMClient(self.config)
        self.parser = ResponseParser()

    def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze user query and return structured information"""
        # Prepare prompt
        prompt = self.templates.query_analysis.format(query=query)

        # Get LLM response
        llm_response = self.llm_client.generate_response(prompt)

        # Parse response
        return self.parser.parse_response(llm_response, original_query=query)