from typing import Dict, Any
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.messages import HumanMessage
from .config import LLMConfig

class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = ChatMistralAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            api_key=self.config.api_key
        )

    def generate_response(self, prompt: str) -> str:
        """Send prompt to LLM and get response"""
        try:
            messages = [HumanMessage(content=prompt)]
            response = self.client.invoke(messages)
            return response.content
        except Exception as e:
            raise Exception(f"LLM API Error: {str(e)}")