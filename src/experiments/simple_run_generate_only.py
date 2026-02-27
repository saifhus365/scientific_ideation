from pathlib import Path
import sys
import json

# Add src to path to allow imports
# This is important if you run this file directly, but the main entry point is run_zeroshot_experiment.py
src_path = Path(__file__).resolve().parents[2]
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from agentic_workflow.state import AgentState
from .zeroshot_graph import generation_only_graph# Assuming this graph exists in simple_graph.py

def run_simple_pipeline(query: str, run_timestamp: str) -> dict:
    """
    Runs a simple, zero-shot idea generation pipeline.
    """
    print(f"  -> Running zero-shot generation for query: '{query[:50]}...'")

    initial_state: AgentState = {
        "initial_query": query,
        "topics": [query], # Use the query as the single topic
        "run_timestamp": run_timestamp,
        "final_ideas": None,
        "final_deduplicated_ideas": None,
        # Set dummy values for other required fields in AgentState if any
        "intention": "To generate novel research ideas.",
        "personalities": [], 
        "persona_pool": [], 
        "history": [],
        "current_round_number": 0, 
        "round_contributions": [],
        "current_criticism": None, 
        "current_summary": None,
        # Ablation flags aren't used but might be expected in the state
        "use_ablation_synthesis": True,
        "use_ablation_RAG": True,
        "use_ablation_viewpoint": True,
        "use_ablation_critique": True,
    }

    # Invoke the graph
    final_agent_state = generation_only_graph.invoke(initial_state)

    # In this simple pipeline, the generated ideas are the final ideas
    # No deduplication is run, so we copy them over.
    final_agent_state["final_deduplicated_ideas"] = final_agent_state.get("final_ideas")

    print(f"  -> Zero-shot generation complete.")
    return final_agent_state

if __name__ == '__main__':
    # Example of how to run it directly
    from datetime import datetime
    
    test_query = "What are the latest advancements in AI for drug discovery?"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    final_state = run_simple_pipeline(test_query, timestamp)
    
    print("\n--- Final State ---")
    print(json.dumps(final_state, indent=2))