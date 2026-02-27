import streamlit as st
from pathlib import Path
from datetime import datetime
import json
import time

# Import the stage functions from your existing main script
from main import (
    run_query_decomposition_stage,
    run_literature_review_stage,
    save_final_results,
    run_paper_download_stage,
    run_paper_processing_stage,
    run_data_indexing_stage,
    run_agentic_workflow_stage,
    run_deduplication_stage,
    state_serializer
)

from agentic_workflow.graph import debate_graph
from agentic_workflow.state import AgentState

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="",
    layout="wide"
)

st.title("AI-Powered Research Assistant")
st.markdown("This application automates the entire research pipeline, from decomposing a query to generating novel hypotheses.")

# --- Main Application Logic ---
project_root = Path.cwd()

# Initialize session state to track progress
if "workflow_complete" not in st.session_state:
    st.session_state.workflow_complete = False
    st.session_state.final_ideas = None
    st.session_state.query_analysis = None
    st.session_state.run_timestamp = None # Initialize the timestamp here

# Input form for the user query
with st.form("research_form"):
    query_text = st.text_area(
        "Enter your research query:", 
        "Explore how ancient trade networks can inform modern blockchain-based supply chains.",
        height=100
    )
    submitted = st.form_submit_button("Start Full Research Workflow")

if submitted:
    st.session_state.workflow_complete = False
    st.session_state.final_ideas = None
    
    # This block will run the entire pipeline sequentially, showing progress updates.
    with st.status("Executing Full Research Workflow...", expanded=True) as status:
        progress_bar = st.progress(0)

        # Stage 1: Query Decomposition
        status.update(label="Stage 1: Decomposing research query...")
        query_analysis_result = run_query_decomposition_stage(project_root, query_text)
        if not query_analysis_result:
            status.update(label="Failed to decompose query. Please try again.", state="error", expanded=True)
            st.stop()
        st.session_state.query_analysis = query_analysis_result
        st.write(f"**Intention:** {query_analysis_result.intention}")
        st.write(f"**Topics:** {', '.join(query_analysis_result.topics)}")
        progress_bar.progress(1/7, text="Stage 1: Query Decomposition Complete")

        # Stage 2: Literature Review
        status.update(label="Stage 2: Conducting literature review...")
        final_papers = run_literature_review_stage(query_analysis_result)
        if not final_papers:
            status.update(label="Literature review did not find any papers.", state="error", expanded=True)
            st.stop()
        st.write(f"Found {len(final_papers)} relevant papers.")
        progress_bar.progress(2/7, text="Stage 2: Literature Review Complete")

        # Generate timestamp for this run and store it in the session state
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state.run_timestamp = run_timestamp
        
        # Save final report
        save_final_results(project_root, query_analysis_result, final_papers, run_timestamp)

        # Stage 3: Paper Downloading
        status.update(label=f"Stage 3: Downloading {len(final_papers)} papers...")
        run_paper_download_stage(final_papers, project_root, run_timestamp)
        st.write("Paper download process complete.")
        progress_bar.progress(3/7, text="Stage 3: Paper Downloading Complete")

        # Stage 4: Paper Processing
        status.update(label="Stage 4: Processing downloaded PDFs...")
        run_paper_processing_stage(project_root, run_timestamp)
        st.write("PDF processing complete.")
        progress_bar.progress(4/7, text="Stage 4: Paper Processing Complete")

        # Stage 5: Data Indexing
        status.update(label="Stage 5: Indexing content into vector database...")
        run_data_indexing_stage(run_timestamp)
        st.write("Data indexing complete.")
        progress_bar.progress(5/7, text="Stage 5: Data Indexing Complete")

        # Stage 6: Agentic Idea Generation
        status.update(label="Stage 6: Running agentic workflow... See details below.")
        
        # Define the initial state for the agentic workflow
        initial_agent_state: AgentState = {
            "initial_query": query_analysis_result.query,
            "topics": query_analysis_result.topics,
            "intention": query_analysis_result.intention,
            "run_timestamp": run_timestamp,
            "use_ablation_synthesis": False, 
            "use_ablation_RAG": False,
            "use_ablation_viewpoint": False,
            "use_ablation_critique": False,
            "personalities": [], "persona_pool": [], "history": [],
            "current_round_number": 0, "round_contributions": [],
            "current_criticism": None, "current_summary": None,
            "final_ideas": None, "final_deduplicated_ideas": None
        }

        # Invoke the graph directly in the app to show live updates
        final_agent_state = debate_graph.invoke(initial_agent_state)
        
        # Save the final state
        state_dir = project_root / "results" / "agent_states"
        state_dir.mkdir(parents=True, exist_ok=True)
        state_path = state_dir / f"workflow_state_{run_timestamp}.json"
        with open(state_path, 'w', encoding='utf-8') as f:
            json.dump(final_agent_state, f, indent=4, default=state_serializer)

        st.write("Agentic workflow complete.")
        progress_bar.progress(6/7, text="Stage 6: Agentic Idea Generation Complete")

        # Stage 7: Deduplicate Final Ideas
        status.update(label="Stage 7: Deduplicating and refining final ideas...")
        final_agent_state = run_deduplication_stage(final_agent_state, project_root, run_timestamp)
        st.write("Deduplication complete.")
        progress_bar.progress(7/7, text="Stage 7: Deduplication and Refinement Complete")


        # Mark workflow as complete
        status.update(label="Workflow finished!", state="complete", expanded=False)
        st.session_state.workflow_complete = True

    # Rerun to display final results outside the status box
    st.rerun()

# Display the final results after the workflow is complete
if st.session_state.workflow_complete:
    st.balloons()
    st.header("Workflow Complete: Final Novel Ideas")
    
    # CORRECTLY retrieve the timestamp from the session state
    run_timestamp = st.session_state.run_timestamp
    if not run_timestamp:
        st.error("Could not find the timestamp for this run. Cannot display results.")
        st.stop()

    state_file = project_root / "results" / "agent_states" / f"workflow_state_{run_timestamp}.json"
    
    try:
        with open(state_file, 'r') as f:
            final_state = json.load(f)
        
        # Get the original and deduplicated idea lists
        original_ideas = final_state.get("final_ideas", {}).get("final_ideas", [])
        deduplicated_ideas_list = final_state.get("final_deduplicated_ideas", {}).get("final_ideas", [])
        
        st.info(f"Displaying {len(deduplicated_ideas_list)} unique ideas (refined from {len(original_ideas)} initial ideas).")

        if deduplicated_ideas_list:
            for i, idea in enumerate(deduplicated_ideas_list):
                with st.expander(f"**Idea {i+1}: {idea.get('title')}**"):
                    st.markdown(f"**Description:** *{idea.get('description')}*")
        else:
            st.warning("Could not extract final ideas from the saved state. The deduplication might have filtered all ideas or an error occurred.")
            
    except FileNotFoundError:
        st.error(f"Could not find the final state file. Looked for: {state_file}")
    except Exception as e:
        st.error(f"An error occurred while reading the final results: {e}")