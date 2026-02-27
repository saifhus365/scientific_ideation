import json
from pathlib import Path
import sys
import numpy as np
import os
from datetime import datetime

# Add the src directory to the Python path to allow for module imports
src_path = Path(__file__).resolve().parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from metrics.deduplication import run_deduplication
# Import the new functions
from metrics.get_precisions import load_and_transform_q1_results, run_external_precisions
from metrics.novelty import calculate_novelty_metrics
from experiments.simple_run import run_simple_workflow


def test_novelty_assessment(deduplicated_ideas, lit_review_report_path):
    """
    Takes a list of deduplicated ideas and the path to the corresponding
    literature review report, then runs the novelty assessment.
    """
    if not deduplicated_ideas:
        print("\n--- Skipping Novelty Assessment: No deduplicated ideas to assess. ---")
        return None, None

    print(f"\n--- Loading literature review report: {lit_review_report_path} ---")
    try:
        with open(lit_review_report_path, 'r', encoding='utf-8') as f:
            lit_review_report = json.load(f)
        discovered_papers = lit_review_report.get("discovered_papers", [])
        if not discovered_papers:
            print("Warning: No 'discovered_papers' found in the report.")
            return None, None
    except FileNotFoundError:
        print(f"Error: The literature review report was not found at {lit_review_report_path}.")
        return None, None
    except json.JSONDecodeError:
        print(f"Error: The literature review report is not a valid JSON file.")
        return None, None

    print("\n--- Running Novelty Assessment ---")
    
    ideas_with_novelty = []
    all_novelty_scores = []
    for idea in deduplicated_ideas:
        novelty_scores = calculate_novelty_metrics(idea, discovered_papers)
        all_novelty_scores.append(novelty_scores)
        idea_with_scores = idea.copy()
        idea_with_scores.update(novelty_scores)
        ideas_with_novelty.append(idea_with_scores)

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
    project_root = Path(__file__).resolve().parent.parent.parent
    
    # --- 1. Load Reference State and External Results ---
    reference_state_path = project_root / "results" / "agent_states" / "workflow_state_20250917_102612.json"
    q_results_path = project_root / "results" / "virsci_results" / "q2_results.json"
    
    with open(reference_state_path, 'r') as f:
        reference_state = json.load(f)
    
    initial_query = reference_state.get("initial_query")
    lit_review_timestamp = reference_state.get("run_timestamp") # For novelty

    print(f"--- Loading and transforming results from: {q_results_path} ---")
    original_ideas = load_and_transform_q1_results(q_results_path)

    # --- 2. Run Simple Workflow for Precision Baseline ---
    if not initial_query:
        raise ValueError("Could not find 'initial_query' in the reference state file.")
    
    # Generate a new timestamp for this evaluation run
    eval_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n--- Running Simple Workflow with query '{initial_query}' for precision baseline ---")
    run_simple_workflow(initial_query, eval_timestamp)

    # --- 3. Run Deduplication ---
    mock_workflow_state = {"final_ideas": {"final_ideas": original_ideas}}
    print("\n--- Running Deduplication on Transformed Ideas ---")
    deduplicated_ideas = run_deduplication(
        final_state=mock_workflow_state,
        project_root_path=project_root,
        run_timestamp=eval_timestamp,
        similarity_threshold=0.8
    )

    if deduplicated_ideas:
        print("\n--- Deduplication Results ---")
        print(f"Original number of ideas: {len(original_ideas)}")
        print(f"Number of ideas after deduplication: {len(deduplicated_ideas)}")
    else:
        print("\nDeduplication did not return any ideas.")
        deduplicated_ideas = [] # Ensure it's a list for the final summary

    # --- 4. Run Precision Evaluation ---
    print(f"\n--- Running External Precision Evaluation against baseline run {eval_timestamp} ---")
    precision_results = run_external_precisions(eval_timestamp, deduplicated_ideas)

    # --- 5. Run Novelty Assessment ---
    novelty_results_detailed, novelty_results_avg = None, None
    if deduplicated_ideas:
        lit_review_path = project_root / "results" / "final_reports" / f"lit_review_report_{lit_review_timestamp}.json"
        novelty_results_detailed, novelty_results_avg = test_novelty_assessment(deduplicated_ideas, lit_review_path)

    # --- 6. Consolidate and save all metrics ---
    q_results_filename = q_results_path.stem
    final_metrics = {
        "run_timestamp": eval_timestamp,
        "source_file": str(q_results_path),
        "deduplication_results": {
            "original_idea_count": len(original_ideas),
            "deduplicated_idea_count": len(deduplicated_ideas),
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

    summary_filepath = summary_dir / f"evaluation_summary_{q_results_filename}_{eval_timestamp}.json"
    with open(summary_filepath, 'w', encoding='utf-8') as f:
        json.dump(final_metrics, f, indent=4)
    
    print(f"\n--- All evaluation metrics for {q_results_filename} saved to: {summary_filepath} ---")