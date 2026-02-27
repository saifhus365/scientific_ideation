from pydantic import BaseModel, Field
from typing import List

class DebaterPersonality(BaseModel):
    """Defines the personality of a debating agent."""
    name: str = Field(..., description="The name of the debater, e.g., 'Dr. Evelyn Reed'.")
    background: str = Field(..., description="The professional background and mindset of the debater.")
    viewpoint: str = Field(..., description="The specific, and potentially controversial, angle the debater will champion.")

class PersonalityList(BaseModel):
    """A container for a list of debater personalities."""
    personalities: List[DebaterPersonality] = Field(..., description="A list of generated debater personas.")

class NovelIdea(BaseModel):
    """Represents a single novel scientific idea or hypothesis."""
    title: str = Field(..., description="A clear and concise title for the idea.")
    description: str = Field(..., description="A detailed description of the idea, its justification, and potential impact.")
    reasoning: str = Field(..., description="The chain of thought or reasoning that led to this idea, from the agent's persona.")

class DebaterContribution(BaseModel):
    """The full contribution of a debater in a single round, including their proposed ideas."""
    debater_name: str = Field(..., description="The name of the debater making the contribution.")
    proposed_ideas: List[NovelIdea] = Field(..., description="A list of novel ideas proposed by the debater in this round.")

class CriticAnalysis(BaseModel):
    """The analysis provided by the critic for a round of debate."""
    critique: str = Field(..., description="The critic's comprehensive analysis of all arguments in the round.")

class RoundSummary(BaseModel):
    """The summary of a debate round, created by the controller."""
    summary: str = Field(..., description="A neutral summary of the key points and conflicts of the round.")

class FinalIdeaList(BaseModel):
    """A container for the final, synthesized list of novel ideas."""
    final_ideas: List[NovelIdea] = Field(..., description="The final, curated list of novel research ideas.")

# --- New: Team selection with reasons ---
class TeamMemberSelection(BaseModel):
    """One selected persona and the reason for selection."""
    persona: DebaterPersonality = Field(..., description="The selected debater persona.")
    reason: str = Field(..., description="Why this persona was selected for the team.")

class TeamSelection(BaseModel):
    """Container for selected team with reasons."""
    selections: List[TeamMemberSelection] = Field(..., description="Exactly the selected personas (e.g., 3) with reasons.")


class FinalIdeaWithAbstract(BaseModel):
    """A final idea, expanded with a full scientific abstract."""
    title: str = Field(description="The clear and concise title of the research idea.")
    abstract: str = Field(description="A full scientific abstract for the research idea, formatted for a paper.")

class FinalIdeaWithAbstractList(BaseModel):
    """A list of the final ideas, each with a full abstract."""
    final_ideas_with_abstracts: List[FinalIdeaWithAbstract]