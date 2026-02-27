import os
from langgraph.graph import StateGraph, END
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv

# Use the main AgentState and data models for compatibility
from agentic_workflow.state import AgentState
from experiments.data_models import FinalIdeaList
from experiments.agent_builders import build_agent

# --- Configuration ---
load_dotenv()
llm = ChatMistralAI(api_key=os.getenv("MISTRAL_API_KEY"), model="mistral-medium-latest", temperature=0.7)

# --- Graph Node Function ---
def generate_ideas_zero_shot(state: AgentState) -> dict:
    """Node to generate 20 novel scientific ideas based on the user query."""
    print("--- 1. GENERATING ZERO-SHOT IDEAS ---")
    idea_generator = build_agent(llm, "simple_idea_generation_prompt", FinalIdeaList)
    
    generated_ideas_list = idea_generator.invoke({
        "initial_query": state["initial_query"],
        "num_ideas": 20
    })
    
    ideas = generated_ideas_list.final_ideas if generated_ideas_list else []
    print(f"Generated {len(ideas)} initial ideas.")
    return {"final_ideas": ideas}

# --- Build the Graph ---
workflow = StateGraph(AgentState)
workflow.add_node("generate_ideas_zero_shot", generate_ideas_zero_shot)
workflow.set_entry_point("generate_ideas_zero_shot")
workflow.add_edge("generate_ideas_zero_shot", END)

# This is the graph that will be imported
generation_only_graph = workflow.compile()
print("Zero-shot idea generation graph compiled successfully.")