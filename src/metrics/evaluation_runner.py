import json
from pathlib import Path
import sys
import numpy as np

# Add the src directory to the Python path
src_path = Path(__file__).resolve().parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from metrics.deduplication import run_deduplication
from metrics.get_precisions import run_precision_evaluation, run_precision_comparison
from metrics.novelty import calculate_novelty_metrics
from experiments.simple_run import run_simple_workflow # Import the function to run the simple agent

def find_simple_run_for_query(project_root: Path, query: str) -> dict | None:
    """Searches simple_agent_states for a run matching the query."""
    simple_agent_states_dir = project_root / "results" / "simple_agent_states"
    if not simple_agent_states_dir.exists():
        return None

    for state_file in sorted(simple_agent_states_dir.glob("*.json"), reverse=True):
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if data.get("initial_query") == query:
                print(f"    -> Found corresponding simple agent run: {state_file.name}")
                return data
        except (json.JSONDecodeError, IOError):
            continue
    return None

def run_full_evaluation(run_timestamp: str, project_root: Path, simple_mode: bool = False, query: str | None = None) -> dict | None:
    """
    Runs the full evaluation suite for a given run_timestamp.
    """
    print(f"  -> Starting evaluation for timestamp: {run_timestamp}")

    if simple_mode:
        state_dir = project_root / "results" / "zeroshot_agent_states"
    else:
        state_dir = project_root / "results" / "agent_states"
        
    final_state_path = state_dir / f"workflow_state_{run_timestamp}.json"
    
    if not final_state_path.exists():
        print(f"    [ERROR] Final state file not found: {final_state_path}")
        return None

    with open(final_state_path, 'r', encoding='utf-8') as f:
        workflow_state = json.load(f)

    precision_results = None
    novelty_results_avg = None

    if simple_mode:
        if not query:
            print("     [ERROR] Query text is required for simple_mode evaluation. Skipping.")
            return {"precision": None, "novelty": None}

        # Try to find the corresponding simple agent run
        simple_run_state = find_simple_run_for_query(project_root, query)
        
        # If not found, run it now and use the result
        if not simple_run_state:
            print(f"     [INFO] No simple agent run found for query. Generating it now...")
            simple_run_state = run_simple_workflow(query, run_timestamp)

        if simple_run_state:
            # 1. Calculate Precision
            zeroshot_ideas = workflow_state.get("final_deduplicated_ideas", [])
            simple_agent_ideas_obj = simple_run_state.get("refined_ideas", {})
            simple_agent_ideas = simple_agent_ideas_obj.get("final_ideas", []) if isinstance(simple_agent_ideas_obj, dict) else []
            precision_results = run_precision_comparison(zeroshot_ideas, simple_agent_ideas, run_timestamp)
        else:
            print(f"     [WARNING] Could not find or generate a simple agent run for query '{query[:30]}...'. Skipping precision.")

        # 2. Handle Novelty
        print("     -> In simple mode, novelty cannot be calculated. Using placeholder.")
        novelty_results_avg = {"average_novelty_score": 0.0}

    else:
        # --- This is the logic for your full pipeline runs ---
        precision_results = run_precision_evaluation(run_timestamp)
        
        deduplicated_ideas_obj = workflow_state.get("final_deduplicated_ideas")
        deduplicated_ideas = deduplicated_ideas_obj.get("final_ideas", []) if isinstance(deduplicated_ideas_obj, dict) else []
        lit_review_path = project_root / "results" / "final_reports" / f"lit_review_report_{run_timestamp}.json"
        
        if deduplicated_ideas and lit_review_path.exists():
            with open(lit_review_path, 'r') as f:
                lit_review_report = json.load(f)
            discovered_papers = lit_review_report.get("discovered_papers", [])
            all_novelty_scores = [calculate_novelty_metrics(idea, discovered_papers) for idea in deduplicated_ideas]
            if all_novelty_scores and all_novelty_scores[0]:
                keys = all_novelty_scores[0].keys()
                novelty_results_avg = {f"average_{key}": np.mean([s[key] for s in all_novelty_scores if s]) for key in keys}

    metrics = {"precision": precision_results, "novelty": novelty_results_avg}
    print(f"  -> Evaluation complete.")
    return metrics