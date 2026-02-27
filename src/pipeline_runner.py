import json
from pathlib import Path
from datetime import datetime

# Import your existing stage functions
from main import (
    run_query_decomposition_stage,
    run_literature_review_stage,
    save_final_results,
    run_paper_download_stage,
    run_paper_processing_stage,
    run_data_indexing_stage,
    run_deduplication_stage,
    state_serializer
)
from agentic_workflow.graph import debate_graph
from agentic_workflow.state import AgentState
from query_decomp.response_parser import QueryAnalysis, Timeline

def run_full_pipeline(query_text: str, ablation_config: dict, project_root: Path) -> str:
    """
    Runs the entire research pipeline for a single query and a given ablation configuration.
    Returns the run_timestamp for this pipeline run.
    """
    print(f"\n{'='*20}\nRunning pipeline for query: '{query_text[:50]}...'")
    print(f"Ablation Config: {ablation_config['name']}")

    if ablation_config.get("use_ablation_query_decomp", False):
        print("  -> SKIPPING Query Decomposition (Ablation)")
        # Create a mock result that uses the raw query text
        query_analysis_result = QueryAnalysis(
            query=query_text,
            topics=[query_text],  # Use the full query as the single topic
            timeline=Timeline(start_date=None, end_date=None, specific_year=None),
            intention="" # Default intention
        )
    else:
        query_analysis_result = run_query_decomposition_stage(project_root, query_text)


    # --- Stages 1-5: Pre-computation ---
    
    if not query_analysis_result:
        print("  -> Failed at Query Decomposition. Skipping.")
        return None

    final_papers = run_literature_review_stage(query_analysis_result)
    if not final_papers:
        print("  -> Failed at Literature Review. Skipping.")
        return None

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    save_final_results(project_root, query_analysis_result, final_papers, run_timestamp)
    run_paper_download_stage(final_papers, project_root, run_timestamp)
    run_paper_processing_stage(project_root, run_timestamp)
    run_data_indexing_stage(run_timestamp)

    # --- Stage 6: Agentic Workflow ---
    initial_agent_state: AgentState = {
        "initial_query": query_analysis_result.query,
        "topics": query_analysis_result.topics,
        "intention": query_analysis_result.intention,
        "run_timestamp": run_timestamp,
        "personalities": [], "persona_pool": [], "history": [],
        "current_round_number": 0, "round_contributions": [],
        "current_criticism": None, "current_summary": None,
        "final_ideas": None, "final_deduplicated_ideas": None,
        # Apply ablation settings from the config dictionary
        "use_ablation_synthesis": ablation_config.get("use_ablation_synthesis", False),
        "use_ablation_RAG": ablation_config.get("use_ablation_RAG", False),
        "use_ablation_viewpoint": ablation_config.get("use_ablation_viewpoint", False),
        "use_ablation_critique": ablation_config.get("use_ablation_critique", False),
    }

    final_agent_state = debate_graph.invoke(initial_agent_state)
    
    # --- Stage 7: Deduplication ---
    final_agent_state = run_deduplication_stage(final_agent_state, project_root, run_timestamp)

    # Save the final state
    state_dir = project_root / "results" / "agent_states"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / f"workflow_state_{run_timestamp}.json"
    with open(state_path, 'w', encoding='utf-8') as f:
        json.dump(final_agent_state, f, indent=4, default=state_serializer)
    
    print(f"  -> Pipeline complete. State saved for timestamp: {run_timestamp}")
    return run_timestamp