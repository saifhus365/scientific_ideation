import json
import os
import random
from collections import defaultdict
from tqdm import tqdm
import retry
from langchain_core.messages import HumanMessage

from metrics.utils import format_plan_json, format_idea_with_abstract

@retry.retry(tries=3, delay=5)
def better_idea(idea_1, idea_2, llm_client, temperature=0.0):
    prompt = (
        "You are a reviewer specialized in Natural Language Processing and Large Language Models. "
        "You are given two research project summaries. One of them is likely to be accepted by a top AI conference (like ICLR or ACL) "
        "and the other one is likely to be rejected. Your task is to identify the one with higher potential.\n\n"
        "The two project proposals are:\n\n"
        "Paper 1:\n"
        f"{format_idea_with_abstract(idea_1)}\n\n"
        "Paper 2:\n"
        f"{format_idea_with_abstract(idea_2)}\n\n"
        "Now, decide which one is the better idea. Directly return a number 1 or 2 and nothing else."
    )
    
    try:
        response = llm_client.invoke([HumanMessage(content=prompt)])
        return prompt, response.content, 0 # Assuming cost is not tracked for Mistral
    except Exception as e:
        print(f"LLM call failed: {e}")
        raise # Re-raise the exception to trigger the retry

def tournament_ranking(idea_lst, llm_client, output_dir, run_timestamp, max_round=3):
    """
    Ranks a list of ideas using a head-to-head tournament evaluation.
    """
    scores = defaultdict(lambda: 1)
    
    for current_round in range(max_round):
        print(f"--- Starting Tournament Round {current_round + 1}/{max_round} ---")
        
        random.shuffle(idea_lst)
        match_pairs = [tuple(idea_lst[i:i+2]) for i in range(0, len(idea_lst) - (len(idea_lst) % 2), 2)]
        
        if len(idea_lst) % 2 != 0:
            scores[format_plan_json(idea_lst[-1])] += 1
        
        for idea1, idea2 in tqdm(match_pairs, desc=f"Round {current_round + 1} Matches"):
            prompt, result, cost = better_idea(idea1, idea2, llm_client)
            
            if result and result.strip() == '1':
                scores[format_plan_json(idea1)] += 1
            else:
                scores[format_plan_json(idea2)] += 1
            
    final_scores = {format_plan_json(idea): scores[format_plan_json(idea)] for idea in idea_lst}
    sorted_ideas = sorted(idea_lst, key=lambda idea: final_scores[format_plan_json(idea)], reverse=True)
    
    ranked_results = []
    for idea in sorted_ideas:
        key = format_plan_json(idea)
        ranked_results.append({
            "title": idea['title'],
            "description": idea['description'],
            "reasoning": idea.get('reasoning'), # Use .get for safety
            "source": idea.get('source'),      # This line preserves the source
            "score": final_scores[key]
        })
        
    output_filename = f"ranked_ideas_{run_timestamp}.json"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, "w") as f:
        json.dump(ranked_results, f, indent=4)
        
    print(f"\nTournament complete. Ranked ideas saved to: {output_path}")
    
    return ranked_results

def calculate_precision_at_n(ranked_ideas, n_values):
    """
    Calculates Precision@N for a given list of N values.
    Assumes each idea in ranked_ideas has a 'source' key.
    """
    results = {}
    print("\n--- Calculating Precision@N ---")
    for n in n_values:
        if len(ranked_ideas) < n:
            print(f"Warning: Not enough ranked ideas to calculate Precision@{n}. Have {len(ranked_ideas)}, need {n}.")
            continue
        
        top_n = ranked_ideas[:n]
        non_baseline_count = sum(1 for idea in top_n if idea.get('source') == 'non_baseline')
        
        precision = non_baseline_count / n
        results[f"Precision@{n}"] = precision
        print(f"Precision@{n}: {precision:.4f} ({non_baseline_count}/{n} from non-baseline)")
        
    return results