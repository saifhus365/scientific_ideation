import json
from pathlib import Path
from scipy.stats import ttest_rel, wilcoxon
import argparse # Use argparse for better command-line flexibility

def find_matching_results(results1: list, results2: list, metric_key: str, n_samples: int) -> tuple[list, list]:
    """
    Finds the first N matching queries between two result lists and returns their metric scores.
    """
    results2_lookup = {res['query']: res for res in results2}
    
    scores1 = []
    scores2 = []
    
    for res1 in results1:
        query = res1['query']
        if query in results2_lookup:
            res2 = results2_lookup[query]
            
            if res1.get('precision', {}).get(metric_key) is not None and \
               res2.get('precision', {}).get(metric_key) is not None:
                
                scores1.append(res1['precision'][metric_key])
                scores2.append(res2['precision'][metric_key])
                
                if len(scores1) >= n_samples:
                    break
                    
    return scores1, scores2

def load_results_data(path: Path, config_key: str) -> list:
    """Loads results data from different JSON structures."""
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Handle the nested structure of live_summary.json
        if "raw_results" in data and config_key in data["raw_results"]:
            return data["raw_results"][config_key]
        # Handle the simpler structure of VirSci results
        elif config_key in data:
            return data[config_key]
        else:
            print(f"Warning: Key '{config_key}' not found in {path}")
            return None
            
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: Could not load or parse {path}. Details: {e}")
        return None
def calculate_average_scores(results: list) -> dict:
    """Calculates the average of each precision metric across a list of results."""
    sums = {}
    counts = {}
    
    for result in results:
        precision_scores = result.get('precision', {})
        for key, value in precision_scores.items():
            if value is not None:
                sums[key] = sums.get(key, 0) + value
                counts[key] = counts.get(key, 0) + 1
                
    averages = {key: sums[key] / counts[key] for key in sums if counts[key] > 0}
    return averages

def main():
    """
    Main function to run significance tests on experiment results.
    Can compare results within the main summary file or against an external file.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    
    # --- Configuration for VirSci comparison ---
    # File 1: Your system's results
    summary_path_1 = project_root / "results" / "experiment_summaries" / "live_summary.json"
    config_1 = "No_Synthesis"
    
    # File 2: The external system's results
    summary_path_2 = project_root / "results" / "experiment_summaries" / "live_summary_VirSci.json"
    config_2 = "VirSci"

    # Test settings
    METRIC_TO_TEST = "Precision@10"
    NUM_SAMPLES = 20 # There are 11 matching queries, we can take the first 10
    
    output_path = project_root / "results" / "experiment_summaries" / f"significance_{config_1}_vs_{config_2}.json"

    results1 = load_results_data(summary_path_1, config_1)
    results2 = load_results_data(summary_path_2, config_2)

    if not results1 or not results2:
        print("Could not load one or both of the result sets. Exiting.")
        return
    # --------------------------------------------
    virsci_averages = calculate_average_scores(results2)
    print(f"--- Average Precision Scores for '{config_2}' ---")
    if virsci_averages:
        # Sort by the 'N' in 'Precision@N' for clean printing
        for key, avg in sorted(virsci_averages.items(), key=lambda item: int(item[0].split('@')[-1])):
            print(f"  -> Average {key}: {avg:.4f}")
    print("-" * 20)
    # -------------------
    print(f"--- Running Significance Test ---")
    print(f"Comparing '{config_1}' from '{summary_path_1.name}'")
    print(f"     vs.  '{config_2}' from '{summary_path_2.name}'")
    print(f"On metric: {METRIC_TO_TEST}\n")

    scores1, scores2 = find_matching_results(
        results1,
        results2,
        METRIC_TO_TEST,
        NUM_SAMPLES
    )

    if len(scores1) < NUM_SAMPLES:
        print(f"Warning: Found only {len(scores1)}/{NUM_SAMPLES} matching queries. A smaller sample size may affect statistical power.")
    
    if len(scores1) < 2: # Need at least 2 samples for a paired test
        print("Error: Not enough matching samples to perform a test. Exiting.")
        return

    print(f"Found {len(scores1)} matching queries for comparison.")

    # Perform statistical tests
    ttest_res = ttest_rel(scores1, scores2)
    wilcoxon_res = wilcoxon(scores1, scores2)
    
    significance_results = {
        f"{config_1}_vs_{config_2}": {
            "metric": METRIC_TO_TEST,
            "num_samples": len(scores1),
            "paired_ttest_pvalue": ttest_res.pvalue,
            "wilcoxon_pvalue": wilcoxon_res.pvalue,
            f"{config_1}_mean_score": sum(scores1) / len(scores1),
            f"{config_2}_mean_score": sum(scores2) / len(scores2)
        },
        f"{config_2}_average_precisions": virsci_averages
    }

    print(f"  -> Avg Score for {config_1}: {significance_results[f'{config_1}_vs_{config_2}'][f'{config_1}_mean_score']:.4f}")
    print(f"  -> Avg Score for {config_2}: {significance_results[f'{config_1}_vs_{config_2}'][f'{config_2}_mean_score']:.4f}")
    print(f"  -> Paired t-test p-value: {ttest_res.pvalue:.4f}")
    print(f"  -> Wilcoxon p-value:    {wilcoxon_res.pvalue:.4f}\n")

    # Save results to a file
    with open(output_path, 'w') as f:
        json.dump(significance_results, f, indent=4)
        
    print(f"--- Significance test results saved to: {output_path} ---")

if __name__ == "__main__":
    main()