import json
from pathlib import Path
import sys
import numpy as np
from datetime import datetime
import shutil

# Add src to path to allow imports
src_path = Path(__file__).resolve().parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from experiments.simple_run_generate_only import run_simple_pipeline
from metrics.evaluation_runner import run_full_evaluation
from pipeline_runner import state_serializer # for saving results

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent
QUERIES_FILE = PROJECT_ROOT / "src" / "test_queries.json"
SUMMARY_DIR = PROJECT_ROOT / "results" / "experiment_summaries"
SUMMARY_PATH = SUMMARY_DIR / "live_summary_zeroshot.json"
AGENT_STATE_DIR = PROJECT_ROOT / "results" / "zeroshot_agent_states"

def summarize_and_save(all_results: dict, output_path: Path):
    """Calculates and prints summary metrics, then saves them along with raw results."""
    
    print(f"\n{'='*20}\n# LIVE EXPERIMENT SUMMARY\n{'='*20}")
    
    final_summary = {}

    for config_name, results_list in all_results.items():
        if not results_list:
            continue

        valid_precisions = [r['precision'] for r in results_list if r.get('precision')]
        valid_novelties = [r['novelty'] for r in results_list if r.get('novelty')]
        avg_p3 = np.mean([p['Precision@3'] for p in valid_precisions if 'Precision@3' in p]) if valid_precisions else 0
        avg_p5 = np.mean([p['Precision@5'] for p in valid_precisions if 'Precision@5' in p]) if valid_precisions else 0
        avg_p10 = np.mean([p['Precision@10'] for p in valid_precisions if 'Precision@10' in p]) if valid_precisions else 0
        avg_p20 = np.mean([p['Precision@20'] for p in valid_precisions if 'Precision@20' in p]) if valid_precisions else 0

        avg_novelty_scores = {}
        if valid_novelties:
            # Ensure there's at least one novelty result before processing
            if valid_novelties[0]:
                first_novelty = valid_novelties[0]
                for key in first_novelty.keys():
                    avg_novelty_scores[key] = np.mean([n[key] for n in valid_novelties if n and key in n])

        final_summary[config_name] = {
            "num_successful_runs": len(results_list),
            "avg_precision_at_3": avg_p3,
            "avg_precision_at_5": avg_p5,
            "avg_precision_at_10": avg_p10,
            "avg_precision_at_20": avg_p20,
            "avg_novelty_scores": avg_novelty_scores
        }
        print(f"\n--- {config_name} ({len(results_list)} successful runs) ---")
        print(f"  Average Precision@3: {avg_p3:.4f}")
        print(f"  Average Precision@5: {avg_p5:.4f}")
        print(f"  Average Precision@10: {avg_p10:.4f}")
        print(f"  Average Precision@20: {avg_p20:.4f}")
        if avg_novelty_scores:
            for key, val in avg_novelty_scores.items():
                print(f"  {key.replace('_', ' ').title()}: {val:.4f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_to_save = {"summary": final_summary, "raw_results": all_results}
    
    with open(output_path, 'w') as f:
        json.dump(data_to_save, f, indent=4)
        
    print(f"\nLive summary and raw results saved to: {output_path}")

def main():
    with open(QUERIES_FILE, 'r') as f:
        queries = json.load(f)["queries"]
    
    print(f"Loaded {len(queries)} test queries.")
    
    all_results = {"ZeroShot": []}
    if SUMMARY_PATH.exists():
        try:
            with open(SUMMARY_PATH, 'r') as f:
                saved_data = json.load(f)
                if "raw_results" in saved_data:
                    all_results = saved_data["raw_results"]
                    print(f"Successfully loaded previous results from {SUMMARY_PATH}")
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load {SUMMARY_PATH}. Starting fresh. Error: {e}")

    for i, query in enumerate(queries):
        print(f"\n{'#'*50}\n# Processing Query {i+1}/{len(queries)}: '{query[:50]}...'\n{'#'*50}")
        
        already_run = any(res.get("query") == query for res in all_results["ZeroShot"])
        if already_run:
            print(f"--- Skipping already completed run for query {i+1} ---")
            continue

        try:
            # 1. Run the zero-shot pipeline
            run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_state = run_simple_pipeline(query, run_timestamp)

            # 2. Save the state so evaluation can find it
            state_path = AGENT_STATE_DIR / f"workflow_state_{run_timestamp}.json"
            state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(state_path, 'w', encoding='utf-8') as f:
                json.dump(final_state, f, indent=4, default=state_serializer)

            # 3. Run evaluation
            metrics = run_full_evaluation(run_timestamp, PROJECT_ROOT, simple_mode=True, query = query)
            if metrics:
                metrics["query"] = query
                all_results["ZeroShot"].append(metrics)
                summarize_and_save(all_results, SUMMARY_PATH)
            
            print(f"--- Completed query {i+1}/{len(queries)} ---")

        except Exception as e:
            print(f"FATAL ERROR on query '{query[:50]}...'. Skipping. Error: {e}")

    print(f"\n\n{'#'*50}\n# FINAL EXPERIMENT SUMMARY\n{'#'*50}")
    summarize_and_save(all_results, SUMMARY_PATH)
    print(f"\nFinal summary is located at: {SUMMARY_PATH}\n{'#'*50}")

if __name__ == "__main__":
    main()