import json
from pathlib import Path
import sys
from dotenv import load_dotenv
import os
import numpy as np

# Add the src directory to the Python path to allow for module imports
src_path = Path(__file__).resolve().parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from metrics.deduplication import run_deduplication
from langchain_mistralai import ChatMistralAI
from metrics.get_precisions import run_precision_evaluation
from metrics.novelty import calculate_novelty_metrics
from KG_explore.modules.semantic_scholar_api import search_papers_by_keyword
from experiments.simple_run import run_simple_workflow



def test_deduplication_on_state_file(run_timestamp):
    """
    Loads a specific workflow state file, runs the deduplication function,
    and prints the results for manual verification.
    """


    print("\n--- Running Deduplication ---")
    
    # Use the deduplicated ideas already in the state if they exist, otherwise run it
    if workflow_state.get("final_deduplicated_ideas"):
        print("Found existing deduplicated ideas in the state file.")
        deduplicated_ideas = workflow_state["final_deduplicated_ideas"].get("final_ideas", [])
    else:
        print("Running deduplication function...")
        deduplicated_ideas = run_deduplication(
            final_state=workflow_state,
            project_root_path=project_root,
            run_timestamp=run_timestamp,
            similarity_threshold=0.8  # You can adjust this threshold for testing
        )

    if deduplicated_ideas is not None:
        print("\n--- Deduplication Results ---")
        original_ideas = workflow_state.get("final_ideas", {}).get("final_ideas", [])
        
        print(f"Original number of ideas: {len(original_ideas)}")
        print(f"Number of ideas after deduplication: {len(deduplicated_ideas)}")
        
        print("\n--- Final Deduplicated Ideas ---")
        for i, idea in enumerate(deduplicated_ideas):
            print(f"{i + 1}. {idea.get('title')}")
    else:
        print("\nDeduplication did not return any ideas. Please check the logs.")
    
    return deduplicated_ideas


def test_evaluation_on_deduplicated_ideas(deduplicated_ideas, run_timestamp):
    """
    Takes a list of deduplicated ideas and runs them through the LLM-as-a-judge
    tournament evaluation using the existing Mistral client.
    """
    if not deduplicated_ideas:
        print("\n--- Skipping Evaluation: No deduplicated ideas to evaluate. ---")
        return

    print("\n--- Running LLM-as-a-Judge Evaluation ---")
    
    load_dotenv()
    llm = ChatMistralAI(api_key=os.getenv("MISTRAL_API_KEY"), model="mistral-medium-latest", temperature=0.0)

    output_dir = Path(__file__).parent / "evaluation_results"
    output_dir.mkdir(exist_ok=True)
    
    tournament_ranking(
        idea_lst=deduplicated_ideas,
        llm_client=llm,
        output_dir=output_dir,
        run_timestamp=run_timestamp,
        max_round=3
    )

def test_novelty_assessment(deduplicated_ideas, lit_review_report_path):
    """
    Takes a list of deduplicated ideas and the path to the corresponding
    literature review report, then runs the novelty assessment.
    """
    if not deduplicated_ideas:
        print("\n--- Skipping Novelty Assessment: No deduplicated ideas to assess. ---")
        return

    print(f"\n--- Loading literature review report: {lit_review_report_path} ---")
    try:
        with open(lit_review_report_path, 'r', encoding='utf-8') as f:
            lit_review_report = json.load(f)
        discovered_papers = lit_review_report.get("discovered_papers", [])
        if not discovered_papers:
            print("Warning: No 'discovered_papers' found in the report.")
            return
    except FileNotFoundError:
        print(f"Error: The literature review report was not found at {lit_review_report_path}.")
        return
    except json.JSONDecodeError:
        print(f"Error: The literature review report is not a valid JSON file.")
        return

    print("\n--- Running Novelty Assessment ---")
    
    ideas_with_novelty = []
    all_novelty_scores = []
    for idea in deduplicated_ideas:
        novelty_scores = calculate_novelty_metrics(idea, discovered_papers)
        all_novelty_scores.append(novelty_scores)
        idea_with_scores = idea.copy()
        idea_with_scores.update(novelty_scores)
        ideas_with_novelty.append(idea_with_scores)


    print("\n--- Novelty Assessment Results ---")
    for idea in ideas_with_novelty:
        print(f"\nTitle: {idea['title']}")
        print(f"  - Historical Dissimilarity (HD): {idea.get('historical_dissimilarity', 0):.4f}")
        print(f"  - Contemporary Dissimilarity (CD): {idea.get('contemporary_dissimilarity', 0):.4f}")
        print(f"  - Contemporary Impact (CI): {idea.get('contemporary_impact', 0):.4f}")
        print(f"  - Overall Novelty (ON): {idea.get('overall_novelty', 0):.4f}")

    avg_novelty = {}
    if all_novelty_scores:
        keys = all_novelty_scores[0].keys()
        for key in keys:
            avg_novelty[f"average_{key}"] = np.mean([s[key] for s in all_novelty_scores])
    
    print("\n--- Average Novelty Scores ---")
    for key, value in avg_novelty.items():
        print(f"  - {key}: {value:.4f}")
        
    return ideas_with_novelty, avg_novelty

if __name__ == "__main__":
    # 1. Run Deduplication
    state_file_path = Path("/Users/husainsaif/Desktop/thesis-saif/results/agent_states/workflow_state_20250929_105341.json")
    project_root = Path("/Users/husainsaif/Desktop/thesis-saif") # Define project root for consistency
    
    print(f"--- Loading state file: {state_file_path} ---")

    try:
        with open(state_file_path, 'r', encoding='utf-8') as f:
            workflow_state = json.load(f)
    except FileNotFoundError:
        print(f"Error: The state file was not found at the specified path.")
    except json.JSONDecodeError:
        print(f"Error: The state file is not a valid JSON file.")

if workflow_state:
        # The timestamp is only needed for saving files, which we will bypass for this test
        initial_query = workflow_state.get("initial_query", "test_run")
        
        run_timestamp = workflow_state.get("run_timestamp", "test_run")
        deduplicated_ideas = test_deduplication_on_state_file(run_timestamp)

        if initial_query:
            simple_workflow_state = run_simple_workflow(initial_query, run_timestamp)
        else:
            print("Warning: Could not find 'initial_query' in state file to run simple workflow.")
            simple_workflow_state = None

        # 2. Run Precision Evaluation
        #run_timestamp_str = "20250821_120933" # Should match the state file
        precision_results = run_precision_evaluation(run_timestamp)

        # 3. Run Novelty Assessment
        novelty_results_detailed, novelty_results_avg = None, None
        if deduplicated_ideas:
            lit_review_path = Path(f"/Users/saif/Desktop/thesis_files/thesis-saif/results/final_reports/lit_review_report_{run_timestamp}.json")
            novelty_results_detailed, novelty_results_avg = test_novelty_assessment(deduplicated_ideas, lit_review_path)

        # 4. Consolidate and save all metrics
        final_metrics = {
            "run_timestamp": run_timestamp,
            "deduplication_results": {
                "original_idea_count": len(workflow_state.get("final_ideas", {}).get("final_ideas", [])),
                "deduplicated_idea_count": len(deduplicated_ideas) if deduplicated_ideas else 0,
                "deduplicated_ideas": deduplicated_ideas
            },
            "precision_results": precision_results,
            "novelty_results": {
                "average_scores": novelty_results_avg,
                "detailed_scores": novelty_results_detailed
            }
        }
        summary_dir = project_root / "results" / "evaluation_summary"
        summary_dir.mkdir(exist_ok=True)

        # Write the final JSON file
        summary_filepath = summary_dir / f"evaluation_summary_{run_timestamp}.json"
        with open(summary_filepath, 'w', encoding='utf-8') as f:
            json.dump(final_metrics, f, indent=4)
        
        print(f"\n--- All evaluation metrics saved to: {summary_filepath} ---")

