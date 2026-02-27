from typing import TypedDict, List, Annotated
from langgraph.graph.message import add_messages
from .data_models import DebaterPersonality, DebaterContribution, CriticAnalysis, RoundSummary, FinalIdeaList, NovelIdea, FinalIdeaWithAbstractList

class AgentState(TypedDict):
    """
    Represents the state of the agentic idea generation workflow.
    """
    # Inputs
    initial_query: str
    topics: List[str]
    intention: str
    run_timestamp: str
    use_ablation_synthesis: bool # Flag to control which synthesis method to use



    # Debate setup
    personalities: List[DebaterPersonality]

    persona_pool: List[DebaterPersonality]
    
    # Full history for final synthesis
    history: Annotated[list, add_messages]
    
    # Round-specific data
    current_round_number: int
    round_contributions: List[DebaterContribution]
    current_criticism: CriticAnalysis
    current_summary: RoundSummary

    # Final output
    final_ideas: FinalIdeaList
    final_ideas_with_abstracts: [FinalIdeaWithAbstractList] # Add this line
    
    final_deduplicated_ideas: [FinalIdeaList]



