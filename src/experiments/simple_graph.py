import os
from langgraph.graph import StateGraph, END
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from typing import List, TypedDict
from experiments.agent_builders import build_agent
from experiments.data_models import FinalIdeaList

# --- Configuration ---
load_dotenv()

# Initialize the LLM
llm = ChatMistralAI(api_key=os.getenv("MISTRAL_API_KEY"), model="mistral-medium-latest", temperature=0.7)

# --- State Definition ---
class SimpleAgentState(TypedDict):
    """A simplified state for the zero-shot idea generation workflow."""
    initial_query: str
    generated_ideas: FinalIdeaList
    refined_ideas: FinalIdeaList

# --- Graph Node Functions ---
def generate_initial_ideas(state: SimpleAgentState) -> dict:
    """Node to generate 20 novel scientific ideas based on the user query."""
    print("--- 1. GENERATING INITIAL IDEAS ---")
    idea_generator = build_agent(llm, "simple_idea_generation_prompt", FinalIdeaList)
    
    initial_ideas = idea_generator.invoke({
        "initial_query": state["initial_query"],
        "num_ideas": 20
    })
    
    print(f"Generated {len(initial_ideas.final_ideas)} initial ideas.")
    return {"generated_ideas": initial_ideas}

def critic_and_refine_ideas(state: SimpleAgentState) -> dict:
    """Node for a critic to evaluate and refine the generated ideas."""
    print("--- 2. CRITIC AND REFINE IDEAS ---")
    critic_agent = build_agent(llm, "simple_critic_prompt", FinalIdeaList)
    
    ideas_text = "\n\n".join(
        [f"- Title: {idea.title}\n  Description: {idea.description}" for idea in state["generated_ideas"].final_ideas]
    )

    refined_ideas = critic_agent.invoke({
        "initial_query": state["initial_query"],
        "ideas_to_refine": ideas_text,
        "num_ideas": 20
    })

    print(f"Critic refined and generated {len(refined_ideas.final_ideas)} ideas.")
    return {"refined_ideas": refined_ideas}

# --- Build the Graph ---
workflow = StateGraph(SimpleAgentState)

workflow.add_node("generate_initial_ideas", generate_initial_ideas)
workflow.add_node("critic_and_refine_ideas", critic_and_refine_ideas)

workflow.set_entry_point("generate_initial_ideas")
workflow.add_edge("generate_initial_ideas", "critic_and_refine_ideas")
workflow.add_edge("critic_and_refine_ideas", END)

simple_graph = workflow.compile()
print("Simple agentic idea generation graph compiled successfully.")
