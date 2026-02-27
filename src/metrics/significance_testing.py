import json
from pathlib import Path
import pandas as pd
import sys
from itertools import combinations
from scipy.stats import ttest_rel, wilcoxon

# Add src to path to allow imports
src_path = Path(__file__).resolve().parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from metrics.statistical_analysis import pairwise_significance

def load_results_from_file(file_path: Path) -> dict:
    """Loads all raw results from a JSON file."""
    if not file_path.exists():
        print(f"Warning: File not found, skipping: {file_path}")
        return {}
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data.get('raw_results', data)
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error loading or parsing {file_path}. Details: {e}")
        return {}

def get_nested_score(result: dict, metric_key: str):
    """
    Extracts a score from a potentially nested dictionary using a dot-separated key.
    Example: "precision.Precision@10" or "novelty.average_overall_novelty"
    """
    keys = metric_key.split('.')
    value = result
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return None
    return value

def main():
    """
    Main function to consolidate experiment results and run pairwise significance tests.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    summaries_dir = project_root / "results" / "experiment_summaries"
    
    # --- Configuration ---
    files_to_load = [
        "live_summary.json",
        "live_summary_no_qdecomp.json"
        # "live_summary_VirSci.json" # VirSci has a different structure for novelty
    ]
    
    # Change this to test different metrics
    # Precision example: "precision.Precision@10"
    # Novelty example: "novelty.average_overall_novelty"
    METRIC_TO_TEST = "novelty.average_overall_novelty"
    
    P_VALUE_THRESHOLD = 0.05
    output_filename = f"pairwise_significance_{METRIC_TO_TEST.replace('.', '_')}.csv"
    output_path = summaries_dir / output_filename
    # ---------------------

    all_raw_results = {}
    for filename in files_to_load:
        results = load_results_from_file(summaries_dir / filename)
        all_raw_results.update(results)
    
    print(f"Loaded results for configurations: {list(all_raw_results.keys())}")

    scores_by_config = {}
    for config_name, results_list in all_raw_results.items():
        if not isinstance(results_list, list): continue
        query_scores = {}
        for result in results_list:
            query = result.get('query')
            if query:
                score = get_nested_score(result, METRIC_TO_TEST)
                if score is not None:
                    query_scores[query] = score
        scores_by_config[config_name] = query_scores

    final_results = {}
    config_names = list(scores_by_config.keys())
    
    print(f"\n--- Performing Detailed Pairwise Significance Tests for '{METRIC_TO_TEST}' ---")

    for config1, config2 in combinations(config_names, 2):
        scores1_map = scores_by_config.get(config1, {})
        scores2_map = scores_by_config.get(config2, {})
        
        common_queries = set(scores1_map.keys()) & set(scores2_map.keys())
        
        if len(common_queries) < 3:
            print(f"\nSkipping '{config1}' vs '{config2}': Found only {len(common_queries)} common queries.")
            continue
            
        list1 = [scores1_map[q] for q in common_queries]
        list2 = [scores2_map[q] for q in common_queries]
        
        analysis_results = pairwise_significance(list1, list2)
        
        comparison_key = f"{config1}_vs_{config2}"
        final_results[comparison_key] = {
            "metric": METRIC_TO_TEST,
            "num_common_queries": len(common_queries),
            "is_significant": bool(analysis_results["p_value"] < P_VALUE_THRESHOLD),
            **analysis_results
        }
        
        print(f"\nComparison: '{config1}' vs '{config2}'")
        print(f"  - Found {len(common_queries)} common queries.")
        print(f"  - Normality of Diffs: {analysis_results['is_normal_on_diffs']}")
        print(f"  - Test Used: {analysis_results['test_used']}")
        print(f"  - P-value: {analysis_results['p_value']:.4f}")
        print(f"  - Effect Size: {analysis_results['effect_size']:.4f}")

    if not final_results:
        print("\nNo comparisons were made. Skipping file save.")
        return
        
    results_df = pd.DataFrame.from_dict(final_results, orient='index')
    results_df.index.name = "comparison"
    results_df.to_csv(output_path)
        
    print(f"\n--- Detailed pairwise significance results saved to: {output_path} ---")

if __name__ == "__main__":
    main()