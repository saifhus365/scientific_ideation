import json
from datetime import datetime
from pathlib import Path
import os
import sys

# Add the src directory to Python path to ensure imports work
src_path = Path(__name__).parent if __name__ != '__main__' else Path(__file__).parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    from query_decomp.query_analyzer import QueryAnalyzer, QueryAnalysis
except ImportError as e:
    print(f"Error: Could not import from 'query_decomp'. {e}")
    print("Make sure it is a package in 'src' and accessible.")
    sys.exit(1)

try:
    from literature_review.agent import LitReviewAgent
except ImportError as e:
    print(f"Error: Could not import from 'literature_review'. {e}")
    print("Make sure it is a package in 'src' and accessible.")
    sys.exit(1)

try:
    from KG_explore.modules.file_io import download_pdf
    from KG_explore.modules.data_processing import sanitize_filename, extract_pdf_url_from_paper_details
except ImportError as e:
    print(f"Error: Could not import download utilities from 'KG_explore'. {e}")
    sys.exit(1)

try:
    from paper_processing.processor import process_directory
except ImportError as e:
    print(f"Error: Could not import from 'paper_processing'. {e}")
    print("Make sure it is a package in 'src' and accessible.")
    sys.exit(1)

try:
    from data_indexing.indexer import index_run
except ImportError as e:
    print(f"Error: Could not import from 'data_indexing'. {e}")
    print("Make sure it is a package in 'src' and accessible.")
    sys.exit(1)

# --- New Imports for Agentic Workflow ---
try:
    from agentic_workflow.graph import debate_graph
    from agentic_workflow.state import AgentState
    from pydantic import BaseModel
    from langchain_core.messages import BaseMessage
except ImportError as e:
    print(f"Error: Could not import from 'agentic_workflow'. {e}")
    print("Make sure it is a package in 'src' and accessible.")
    sys.exit(1)

try:
    from metrics.deduplication import run_deduplication
except ImportError as e:
    print(f"Error: Could not import from 'metrics'. {e}")
    sys.exit(1)


def state_serializer(obj):
    """Custom JSON serializer for objects in the agent state."""
    if isinstance(obj, BaseModel):
        return obj.dict()
    if isinstance(obj, BaseMessage):
        return obj.dict()
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_final_results(project_root: Path, query_analysis: QueryAnalysis, paper_list: list, timestamp: str) -> Path:
    """Saves the final results of the entire workflow to a single JSON file."""
    results_dir = project_root / "results" / "final_reports"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    output_filepath = results_dir / f"lit_review_report_{timestamp}.json"
    
    report = {
        "initial_query": query_analysis.query,
        "query_analysis": {
            "topics": query_analysis.topics,
            "timeline": {
                "start_date": query_analysis.timeline.start_date,
                "end_date": query_analysis.timeline.end_date,
                "specific_year": query_analysis.timeline.specific_year,
            },
            "intention": query_analysis.intention,
        },
        "discovered_papers": paper_list,
    }
    
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4)
        
    print(f"\nFinal report saved to: {output_filepath}")
    return output_filepath

def run_query_decomposition_stage(project_root_path: Path, query_text: str) -> QueryAnalysis:
    """Handles the query decomposition stage."""
    print("\n--- Stage 1: Query Decomposition ---")
    analyzer = QueryAnalyzer() 
    
    # REMOVED: query_text = input('Enter your research query: ')
    
    try:
        analysis_result = analyzer.analyze_query(query_text)
        analysis_result.query = query_text # Attach the original query for later use
        
        print("\nQuery Analysis Results:")
        print(f"  Topics: {analysis_result.topics}")
        print(f"  Timeline: Start={analysis_result.timeline.start_date}, End={analysis_result.timeline.end_date}, Specific={analysis_result.timeline.specific_year}")
        print(f"  User Intention: {analysis_result.intention}")
        
        return analysis_result
        
    except Exception as e:
        print(f"Error during query analysis: {str(e)}")
        return None

def run_literature_review_stage(query_analysis: QueryAnalysis) -> list:
    """Runs the agentic literature review process."""
    print("\n--- Stage 2: Agentic Literature Review ---")
    if not query_analysis.topics:
        print("No topics found from query analysis. Cannot start literature review.")
        return []
    
    # Combine all topics to form the seed for the agent
    initial_topic = " and ".join(query_analysis.topics)
    print(f"Starting literature review with combined topic: '{initial_topic}'")
    agent = LitReviewAgent(initial_topic)
    
    try:
        final_paper_list = agent.run()
        print(f"\nLiterature review complete. Found {len(final_paper_list)} unique papers.")
        return final_paper_list
    except Exception as e:
        print(f"An error occurred during the literature review: {e}")
        return []

def run_paper_download_stage(papers_to_download: list, project_root_path: Path, timestamp: str):
    """Handles the downloading of PDFs from the final paper list."""
    print("\n--- Stage 3: Paper Downloading ---")
    
    if not papers_to_download:
        print("No papers to download. Skipping.")
        return

    # Create a dedicated download directory named after the report timestamp
    download_folder_name = f"lit_review_papers_{timestamp}"
    download_dir = project_root_path / "data" / "papers" / download_folder_name
    
    print(f"Saving papers to: {download_dir}\n")

    for i, paper in enumerate(papers_to_download):
        title = paper.get("title", "Unknown Title")
        paper_id = paper.get("paperId", f"unknown_id_{i}")
        
        # Use the more robust URL extraction logic
        pdf_url = extract_pdf_url_from_paper_details(paper)

        if pdf_url:
            download_pdf(
                title=title,
                paper_id=paper_id,
                pdf_url=pdf_url,
                download_dir=str(download_dir),
                sanitize_func=sanitize_filename
            )
        else:
            print(f"    -> No open access PDF URL for paper {paper_id} ('{title}'). Skipping.")

    print("\n--- Download process finished. ---")


def run_paper_processing_stage(project_root_path: Path, timestamp: str):
    """Runs the PDF processing stage on the downloaded papers."""
    print("\n--- Stage 4: Paper Processing ---")
    
    download_folder_name = f"lit_review_papers_{timestamp}"
    input_dir = project_root_path / "data" / "papers" / download_folder_name
    output_dir = project_root_path / "data" / "processed_papers" / download_folder_name
    
    if not input_dir.exists() or not any(input_dir.iterdir()):
        print(f"Input directory '{input_dir}' is empty or does not exist. Skipping processing.")
        return
        
    print(f"Processing PDFs from: {input_dir}")
    print(f"Saving JSON output to: {output_dir}")
    
    process_directory(input_pdf_dir=input_dir, output_json_dir=output_dir)

    print("\n--- Paper processing finished. ---")


def run_data_indexing_stage(run_timestamp: str):
    """Runs the data indexing workflow for a specific run."""
    print("\n--- Stage 5: Data Indexing ---")
    try:
        index_run(run_timestamp)
    except Exception as e:
        print(f"An error occurred during data indexing: {e}")

# --- New Stage for Agentic Workflow ---

def state_serializer(obj):
    """Custom JSON serializer for objects in the agent state."""
    if isinstance(obj, BaseModel):
        return obj.dict()
    if isinstance(obj, BaseMessage):
        return obj.dict()
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def run_agentic_workflow_stage(query_analysis: QueryAnalysis, run_timestamp: str, project_root_path: Path):
    """Runs the agentic idea generation workflow."""
    print("\n--- Stage 6: Agentic Idea Generation ---")

    # 1. Define the initial state using data from previous stages
    initial_state: AgentState = {
        "initial_query": query_analysis.query,
        "topics": query_analysis.topics,
        "intention": query_analysis.intention,
        "run_timestamp": run_timestamp,
        "personalities": [],
        "persona_pool": [],
        "history": [],
        "current_round_number": 0,
        "round_contributions": [],
        "current_criticism": None,
        "current_summary": None,
        "final_ideas": None,
        'final_deduplicated_ideas': None
    }
    print(f"Starting agentic workflow for query: '{initial_state['initial_query']}'")

    # 2. Run the graph
    final_state = debate_graph.invoke(initial_state)

    # 3. View the final list of novel ideas
    print("\n\n--- FINAL NOVEL IDEAS ---")
    if final_state.get("final_ideas"):
        for i, idea in enumerate(final_state["final_ideas"].final_ideas):
            print(f"{i+1}. {idea.title}")
            print(f"   Description: {idea.description}\n")
    else:
        print("No final ideas were generated by the agentic workflow.")

    # 4. Save the final state to a JSON file
    try:
        print("\n--- Saving Final Workflow State ---")
        state_dir = project_root_path / "results" / "agent_states"
        state_dir.mkdir(parents=True, exist_ok=True)
        
        state_path = state_dir / f"workflow_state_{run_timestamp}.json"
        
        with open(state_path, 'w', encoding='utf-8') as f:
            json.dump(final_state, f, indent=4, default=state_serializer)
            
        print(f"Final agent state saved successfully to: {state_path}")

    except Exception as e:
        print(f"\nError saving final agent state: {e}")
    
    print("\n--- Agentic idea generation finished. ---")

    return final_state

def run_deduplication_stage(final_agent_state: dict, project_root_path: Path, run_timestamp: str):
    """Runs the deduplication process on the final generated ideas."""
    print("\n--- Stage 7: Deduplicating Final Ideas ---")
    if not final_agent_state:
        print("Skipping deduplication because the agentic workflow did not produce a final state.")
        return None
    
    try:
        deduplicated_ideas_list = run_deduplication(
            final_state=final_agent_state,
            project_root_path=project_root_path,
            run_timestamp=run_timestamp,
            similarity_threshold=0.80 # Using a threshold of 80%
        )
        if deduplicated_ideas_list is not None:
            # Add the deduplicated ideas to the final state
            final_agent_state["final_deduplicated_ideas"] = {"final_ideas": deduplicated_ideas_list}

            # Re-save the updated state file
            state_dir = project_root_path / "results" / "agent_states"
            state_path = state_dir / f"workflow_state_{run_timestamp}.json"
            
            with open(state_path, 'w', encoding='utf-8') as f:
                json.dump(final_agent_state, f, indent=4, default=state_serializer)
            
            print(f"Successfully updated agent state with deduplicated ideas: {state_path}")
            return final_agent_state
        
        return final_agent_state
    except Exception as e:
        print(f"An error occurred during deduplication: {e}")
        return None



if __name__ == "__main__":
    project_root = Path.cwd() 
    print(f"Executing workflow. Project root (CWD): {project_root}")

        # Define a default query for standalone execution
    default_query = "Science Search And research paper retrieval methodologies using Multi Agent Systems"
    print(f"Running with default query: '{default_query}'")

    # Stage 1: Query Decomposition
    query_analysis_result = run_query_decomposition_stage(project_root, default_query)

    if query_analysis_result:
        # Stage 2: Literature Review
        final_papers = run_literature_review_stage(query_analysis_result)
        
        if final_papers:
            # Generate a single timestamp for this run
            run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save the final report
            save_final_results(project_root, query_analysis_result, final_papers, run_timestamp)

            # Stage 3: Download Papers from the report
            run_paper_download_stage(final_papers, project_root, run_timestamp)
            
            # Stage 4: Process Downloaded Papers
            run_paper_processing_stage(project_root, run_timestamp)

            # Stage 5: Index Processed Papers
            run_data_indexing_stage(run_timestamp)

            # Stage 6: Run Agentic Idea Generation Workflow
            #final_agent_state = run_agentic_workflow_stage(query_analysis_result, run_timestamp, project_root)
            
            # --- New Stage 7: Deduplicate Final Ideas ---
        else:
            print("\nLiterature review did not yield any papers.")
    else:
        print("\nQuery decomposition failed. Skipping subsequent stages.")

    print("\n" + "="*30 + " COMBINED WORKFLOW FINISHED " + "="*30)