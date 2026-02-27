import json
from pathlib import Path
import sys
from dotenv import load_dotenv
import os

# Add the src directory to the Python path to allow for module imports
src_path = Path(__file__).resolve().parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from metrics.deduplication import run_deduplication
from langchain_mistralai import ChatMistralAI

from metrics.llm_evaluation import tournament_ranking, calculate_precision_at_n


def run_precision_comparison(baseline_ideas: list, non_baseline_ideas: list, run_timestamp: str) -> dict | None:
    """
    Ranks ideas from two lists via tournament and calculates Precision@N.
    This version is flexible and takes idea lists directly.
    """
    # 1. Setup LLM and paths
    load_dotenv()
    llm = ChatMistralAI(api_key=os.getenv("MISTRAL_API_KEY"), model="mistral-medium-latest", temperature=0.0)
    output_dir = Path(__file__).parent / "evaluation_results"
    output_dir.mkdir(exist_ok=True)

    # 2. Tag ideas with their source
    for idea in baseline_ideas:
        idea['source'] = 'baseline'
    for idea in non_baseline_ideas:
        idea['source'] = 'non_baseline'

    # 3. Combine ideas and run tournament
    all_ideas = baseline_ideas + non_baseline_ideas
    if not all_ideas:
        print("    [WARNING] No ideas provided for precision comparison.")
        return None
        
    print(f"    -> Running precision tournament for {len(all_ideas)} ideas...")
    
    ranked_ideas = tournament_ranking(
        idea_lst=all_ideas,
        llm_client=llm,
        output_dir=output_dir,
        run_timestamp=run_timestamp,
        max_round=10
    )
    # 4. Calculate and return Precision@N
    if ranked_ideas:
        n_values = [3, 5, 10, 20]
        precision_results = calculate_precision_at_n(ranked_ideas, n_values)
        return precision_results
    return None



def run_precision_evaluation(timestamp_str):
    """
    Loads ideas from baseline and non-baseline configurations,
    ranks them via tournament, and calculates Precision@N.
    """
    # 1. Setup paths and load LLM
    load_dotenv()
    llm = ChatMistralAI(api_key=os.getenv("MISTRAL_API_KEY"), model="mistral-medium-latest", temperature=0.0)
    project_root = Path("/Users/husainsaif/Desktop/thesis-saif")
    
    baseline_path = project_root / "results" / "simple_agent_states" / f"simple_workflow_state_{timestamp_str}.json"
    non_baseline_path = project_root / "results" / "agent_states" / f"workflow_state_{timestamp_str}.json"


    try:
        run_timestamp = non_baseline_path.stem.split('workflow_state_')[-1]
    except IndexError:
        print("Could not extract timestamp from filename. Using a default.")
        run_timestamp = "default_timestamp"

    # 2. Load ideas from files
    try:
        with open(baseline_path, 'r', encoding='utf-8') as f:
            baseline_data = json.load(f)
        baseline_ideas = baseline_data.get("refined_ideas", {}).get("final_ideas", [])

        with open(non_baseline_path, 'r', encoding='utf-8') as f:
            non_baseline_data = json.load(f)
        non_baseline_ideas = non_baseline_data.get("final_deduplicated_ideas", {}).get("final_ideas", [])

        print(f"Loaded {len(baseline_ideas)} baseline ideas.")
        print(f"Loaded {len(non_baseline_ideas)} non-baseline ideas.")

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading idea files: {e}")
        return

    # 3. Tag ideas with their source
    for idea in baseline_ideas:
        idea['source'] = 'baseline'
    for idea in non_baseline_ideas:
        idea['source'] = 'non_baseline'

    # 4. Combine ideas into a single pool
    all_ideas = baseline_ideas + non_baseline_ideas
    print(f"Total ideas for tournament: {len(all_ideas)}")

    # 5. Run tournament ranking
    output_dir = Path(__file__).parent / "evaluation_results"
    output_dir.mkdir(exist_ok=True)
    
    ranked_ideas = tournament_ranking(
        idea_lst=all_ideas,
        llm_client=llm,
        output_dir=output_dir,
        run_timestamp=run_timestamp,
        max_round=10  # As per your requirement for 10 rounds
    )

    # 6. Calculate and print Precision@N
    if ranked_ideas:
        n_values = [3, 5, 10, 20]
        precision_results = calculate_precision_at_n(ranked_ideas, n_values)
        return precision_results

