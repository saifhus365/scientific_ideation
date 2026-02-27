import json
from pathlib import Path
from pydantic import BaseModel
from langchain_core.messages import BaseMessage
from .graph import debate_graph
from .state import AgentState

# This file provides an example of how to run the agentic debate workflow.

# --- Helper for saving state ---
def state_serializer(obj):
    """Custom JSON serializer for objects in the agent state."""
    if isinstance(obj, BaseModel):
        return obj.dict()
    if isinstance(obj, BaseMessage):
        # BaseMessage objects are also Pydantic models, but this is explicit
        return obj.dict()
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

# 1. Define your inputs
initial_state: AgentState = {
    "initial_query": "Science Search And research paper retrieval methodologies using Multi Agent Systems",
    "topics": [
            "Science Search",
            "Research Paper Retrieval",
            "Multi Agent Systems"
        ],
    "intention": "Descriptive",
    "run_timestamp": "20250811_112228",# IMPORTANT: Use a real timestamp from your runs
    "personalities": [],
    "history": [],
    "current_round_number": 0,
    "round_contributions": [],
    "current_criticism": None,
    "current_summary": None,
    "final_ideas": None,
}

# 2. Run the graph
final_state = debate_graph.invoke(initial_state)

# 3. View the final list of novel ideas
print("\n\n--- FINAL NOVEL IDEAS ---")
if final_state.get("final_ideas"):
    for i, idea in enumerate(final_state["final_ideas"].final_ideas):
        print(f"{i+1}. {idea.title}")
        print(f"   Description: {idea.description}\n")
else:
    print("No final ideas were generated.")

# 4. Save the final state to a JSON file
try:
    print("\n--- Saving Final Workflow State ---")
    state_dir = Path(__file__).resolve().parent.parent.parent / "results" / "agent_states"
    state_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = final_state.get("run_timestamp", "unknown_run")
    state_path = state_dir / f"workflow_state_{timestamp}.json"
    
    with open(state_path, 'w', encoding='utf-8') as f:
        # Use the custom serializer to handle Pydantic models and other objects
        json.dump(final_state, f, indent=4, default=state_serializer)
        
    print(f"Final state saved successfully to: {state_path}")

except Exception as e:
    print(f"\nError saving final state: {e}")