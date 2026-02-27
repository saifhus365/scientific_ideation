from pydantic import BaseModel, Field
from typing import List




class NovelIdea(BaseModel):
    """Represents a single novel scientific idea or hypothesis."""
    title: str = Field(..., description="A clear and concise title for the idea.")
    description: str = Field(..., description="A detailed description of the idea, its justification, and potential impact.")
    reasoning: str = Field(..., description="The chain of thought or reasoning that led to this idea, from the agent's persona.")

class FinalIdeaList(BaseModel):
    """A container for the final, synthesized list of novel ideas."""
    final_ideas: List[NovelIdea] = Field(..., description="The final, curated list of novel research ideas.")
