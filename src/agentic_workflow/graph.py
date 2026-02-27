import os
from pathlib import Path
import stat
from langgraph.graph import StateGraph, END
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
import streamlit as st

from .state import AgentState
from .agent_builders import build_structured_agent, build_raw_agent, get_retriever_tool
from .data_models import PersonalityList, DebaterContribution, CriticAnalysis, RoundSummary, FinalIdeaList, TeamSelection, FinalIdeaWithAbstractList, FinalIdeaWithAbstract

# --- Configuration ---
load_dotenv()
MAX_DEBATE_ROUNDS = 3
NUM_DEBATERS = 3
PERSONA_POOL_MIN = 5
PERSONA_POOL_MAX = 10

# Initialize the LLM
llm = ChatMistralAI(api_key=os.getenv("MISTRAL_API_KEY"), model="mistral-medium-latest", temperature = 0.7)

# --- Graph Node Functions ---

def generate_persona_pool(state: AgentState) -> dict:
    """Node to generate a pool of candidate personas."""
    print("--- 1a. CONTROLLER: GENERATING PERSONA POOL ---")
    pool_generator = build_structured_agent(llm, "persona_pool_prompt", PersonalityList)
    persona_pool = pool_generator.invoke({
        "initial_query": state["initial_query"],
        "intention": state["intention"],
        "topics": ", ".join(state["topics"]),
        "min_count": PERSONA_POOL_MIN,
        "max_count": PERSONA_POOL_MAX
    })
    print(f"Generated a pool of {len(persona_pool.personalities)} personas.")
    return {"persona_pool": persona_pool.personalities}

def select_team_from_pool(state: AgentState) -> dict:
    """Node to select the final team from the persona pool."""
    print("--- 1b. CONTROLLER: SELECTING FINAL TEAM ---")
    team_selector = build_structured_agent(llm, "team_selection_prompt", TeamSelection)
    
    candidate_list_str = "\n".join(
        f"- Name: {p.name}\n  Background: {p.background}\n  Viewpoint: {p.viewpoint}" 
        for p in state["persona_pool"]
    )
    
    team_selection = team_selector.invoke({
        "intention": state["intention"],
        "topics": ", ".join(state["topics"]),
        "num_debaters": NUM_DEBATERS,
        "candidate_list": candidate_list_str
    })
    
    final_personalities = [selection.persona for selection in team_selection.selections]
    
    print("Selected Team:")
    for selection in team_selection.selections:
        print(f"  - {selection.persona.name}: {selection.reason}")
        
    return {
        "personalities": final_personalities, 
        "current_round_number": 1, 
        "history": []
    }

def idea_generation_round(state: AgentState) -> dict:
    """Node for a round of idea generation from each scientist in a conversational chain."""
    print(f"\n--- 2. STARTING IDEA GENERATION ROUND {state['current_round_number']} ---")
    use_ablation = state.get("use_ablation_synthesis", False)
    use_ablation1 = state.get("use_ablation_viewpoint",False)
    use_abltation2 = state.get("use_ablation_RAG",False)
    if use_ablation:
        print("--- USING ABLATION FOR IDEA GENERATION ---")
        prompt_key = "ablation_idea_generation_agent_prompt"
    else:
        prompt_key = "idea_generation_agent_prompt"

    summary_text = state["current_summary"].summary if state.get("current_summary") else "This is the first round."
    retriever = get_retriever_tool(state["run_timestamp"])
    if use_ablation:
        idea_agent = build_structured_agent(llm, "ablation_idea_generation_agent_prompt", DebaterContribution)
    else:
        idea_agent = build_structured_agent(llm, "idea_generation_agent_prompt", DebaterContribution)
    
    round_contributions = []
    previous_contributions_text = "No one has contributed yet in this round. You are the first."

    for persona in state["personalities"]:
        with st.expander(f"**🔬 Scientist: {persona.name} ({persona.background})**"):
            st.write(f"**Viewpoint:** *{persona.viewpoint}*")
            st.write("---")
            st.info("Generating ideas based on the previous round's summary and new context...")
            
            if use_abltation2:
                print("--- SKIPPING RETRIEVAL (RAG ABLATION) ---")
                context = ""
            else:
                if use_ablation1:
                    print("--- USING ABLATION RETRIEVAL QUERY (Initial Query Only) ---")
                    retriever_query = state['initial_query']
                else:
                    retriever_query = f"{state['initial_query']} from the perspective of {persona.viewpoint}"
                context = retriever(retriever_query)


            
            response = idea_agent.invoke({
                "persona_name": persona.name, "persona_background": persona.background,
                "persona_viewpoint": persona.viewpoint, "initial_query": state["initial_query"],
                "round_summary": summary_text, "context": context,
                "previous_contributions": previous_contributions_text
            })
            
            round_contributions.append(response)
            st.success(f"Generated {len(response.proposed_ideas)} new ideas.")

            for idea in response.proposed_ideas:
                st.markdown(f"**Idea:** {idea.title}")
                st.markdown(f"**Description:** {idea.description}")
                st.markdown(f"**Reasoning:** *{idea.reasoning}*")
            
            # Format this agent's contribution to be passed to the next agent
            new_contribution_text = f"Contribution from {response.debater_name}:\n"
            for idea in response.proposed_ideas:
                new_contribution_text += f"- Idea: {idea.title}\n  - Description: {idea.description}\n  - Reasoning: {idea.reasoning}\n"
            
            if previous_contributions_text == "No one has contributed yet in this round. You are the first.":
                previous_contributions_text = new_contribution_text
            else:
                previous_contributions_text += "\n\n" + new_contribution_text
        
    return {"round_contributions": round_contributions}


def critic_round(state: AgentState) -> dict:
    """Node for the critic to analyze the generated ideas."""
    st.write("### ⚖️ Critic's Turn")
    with st.expander("**Impartial Critic Evaluating the Round's Ideas**"):
        st.info("The critic is analyzing the novelty, feasibility, and potential impact of all proposed ideas...")
        critic = build_structured_agent(llm, "critic_agent_prompt", CriticAnalysis)
        ideas_text = "\n\n".join(
            f"Ideas from {contrib.debater_name}:\n" +
            "\n".join([f"- Title: {idea.title}\n  Description: {idea.description}\n  Reasoning: {idea.reasoning}" for idea in contrib.proposed_ideas])
            for contrib in state["round_contributions"]
        )
        criticism = critic.invoke({"proposed_ideas": ideas_text})
        st.success("Critic's analysis is complete.")
        st.markdown(criticism.critique)
    return {"current_criticism": criticism}


def summarize_round(state: AgentState) -> dict:
    """Node for the controller to summarize the round for reflection."""
    st.write("### 📝 Round Summary")

    use_ablation3 = state.get("use_ablation_critique", False)

    with st.expander("**Moderator Summarizing the Round**"):
        # The misleading st.info and redundant agent creation have been removed from here.
        ideas_text = "\n\n".join(
            f"Ideas from {contrib.debater_name}:\n" +
            "\n".join([f"- Title: {idea.title}\n  Description: {idea.description}\n  Reasoning: {idea.reasoning}" for idea in contrib.proposed_ideas])
            for contrib in state["round_contributions"]
        )
        
        if use_ablation3:
            # This message is now correctly shown only during the ablation run.
            st.info("The moderator is creating a concise summary of the ideas (criticism skipped)...")
            prompt_key = "ablation_round_summary_prompt"
            summary_input = {"proposed_ideas": ideas_text}
        else:
            # This message is shown when the critic is active.
            st.info("The moderator is creating a concise summary of the ideas and the critic's feedback...")
            prompt_key = "round_summary_prompt"
            summary_input = {
                "proposed_ideas": ideas_text,
                "criticism": state["current_criticism"].critique
            }

        summarizer = build_structured_agent(llm, prompt_key, RoundSummary)
        summary = summarizer.invoke(summary_input)

        st.success("Round summary is complete.")
        st.markdown(summary.summary)
        
    history_message = f"**Round {state['current_round_number']} Summary:**\n{summary.summary}"
    return {
        "current_summary": summary,
        "current_round_number": state["current_round_number"] + 1,
        "history": [("system", history_message)]
    }

def generate_abstracts(state: AgentState) -> dict:
    """Node to take final ideas and generate a full scientific abstract for each."""
    print("\n--- 6. ABSTRACT GENERATOR: CREATING SCIENTIFIC ABSTRACTS ---")
    if not state.get("final_ideas") or not state["final_ideas"].final_ideas:
        print("  -> No final ideas to process. Skipping abstract generation.")
        # Return a properly structured empty list to avoid downstream errors
        return {"final_ideas_with_abstracts": FinalIdeaWithAbstractList(final_ideas_with_abstracts=[])}

    # Build an agent that returns a structured string, not a JSON object
    abstractor_agent = build_raw_agent(llm, "abstract_generation_prompt")
    
    ideas_with_abstracts = []
    final_ideas_list = state["final_ideas"].final_ideas
    
    with st.status("Generating final abstracts for top ideas...", expanded=True) as status:
        for i, idea in enumerate(final_ideas_list):
            status.update(label=f"Generating abstract for idea {i+1}/{len(final_ideas_list)}: '{idea.title}'")
            
            # Invoke the agent to generate the abstract as a raw string
            response_text = abstractor_agent.invoke({
                "idea_title": idea.title,
                "idea_description": idea.description
            })
            
            # Create the new data model instance
            idea_with_abstract = FinalIdeaWithAbstract(title=idea.title, abstract=response_text)
            ideas_with_abstracts.append(idea_with_abstract)

    print(f"  -> Abstract generation complete.")
    return {"final_ideas_with_abstracts": FinalIdeaWithAbstractList(final_ideas_with_abstracts=ideas_with_abstracts)}


def synthesize_final_ideas(state: AgentState) -> dict:
    """Node for the final synthesizer to create the list of novel ideas."""
    use_ablation = state.get("use_ablation_synthesis", False)

    if use_ablation:
        print("\n--- 5. ABLATION SYNTHESIZER: CREATING FINAL IDEA LIST FROM FINAL ROUND ---")
        prompt_key = "ablation_final_synthesis_prompt"
        
        # Format the ideas from the final round
        ideas_text = "\n\n".join(
            f"Ideas from {contrib.debater_name}:\n" +
            "\n".join([f"- Title: {idea.title}\n  Description: {idea.description}\n  Reasoning: {idea.reasoning}" for idea in contrib.proposed_ideas])
            for contrib in state["round_contributions"]
        )
        
        # Get the critic's analysis from the final round
        criticism_text = state["current_criticism"].critique
        
        synthesizer = build_structured_agent(llm, prompt_key, FinalIdeaList)
        final_ideas = synthesizer.invoke({
            "final_round_ideas": ideas_text,
            "final_criticism": criticism_text
        })

    else:
        print("\n--- 5. SYNTHESIZER: CREATING FINAL IDEA LIST FROM HISTORY ---")
        prompt_key = "final_synthesis_prompt"
        synthesizer = build_structured_agent(llm, prompt_key, FinalIdeaList)
        full_history = "\n\n".join([
            msg.content for msg in state.get('history', []) if msg.type == 'system'
        ])
        final_ideas = synthesizer.invoke({"history": full_history})

    print(f"  -> Final synthesis complete. Generated {len(final_ideas.final_ideas)} novel ideas.")
    return {"final_ideas": final_ideas}



def should_continue(state: AgentState) -> str:
    """Conditional edge to decide whether to continue the debate or end."""
    if state["current_round_number"] > MAX_DEBATE_ROUNDS:
        return "end"
    return "continue"


def should_criticize(state: AgentState) -> str:
    """Conditional edge to decide whether to run the critic round."""
    if state.get("use_ablation_critique", False):
        print("--- SKIPPING CRITIC ROUND (CRITIQUE ABLATION) ---")
        return "summarize"
    return "criticize"

# --- Build the Graph ---
workflow = StateGraph(AgentState)
workflow.add_node("generate_persona_pool", generate_persona_pool)
workflow.add_node("select_team_from_pool", select_team_from_pool)
workflow.add_node("idea_generation_round", idea_generation_round)
workflow.add_node("critic_round", critic_round)
workflow.add_node("summarize_round", summarize_round)
workflow.add_node("synthesize_final_ideas", synthesize_final_ideas)
# Add the new node


workflow.set_entry_point("generate_persona_pool")
workflow.add_edge("generate_persona_pool", "select_team_from_pool")
workflow.add_edge("select_team_from_pool", "idea_generation_round")
workflow.add_conditional_edges(
    "idea_generation_round",
    should_criticize,
    {
        "criticize": "critic_round",
        "summarize": "summarize_round",
    },
)

workflow.add_edge("critic_round", "summarize_round")
workflow.add_conditional_edges(
    "summarize_round", should_continue,
    {"continue": "idea_generation_round", "end": "synthesize_final_ideas"}
)
workflow.add_edge("synthesize_final_ideas", END) # Route to the new node
 

debate_graph = workflow.compile()

debate_graph = workflow.compile()
print("Agentic idea generation graph compiled successfully.")

# --- Save Graph Visualization ---
try:
    # Define the output directory and ensure it exists
    viz_dir = Path(__file__).resolve().parent.parent.parent / "results" / "graph_visuals"
    viz_dir.mkdir(parents=True, exist_ok=True)
    

    # --- Option 2: Save Mermaid Markdown File ---
    mermaid_path = viz_dir / "agentic_workflow_graph.md"
    mermaid_syntax = debate_graph.get_graph().draw_mermaid()
    with open(mermaid_path, "w") as f:
        f.write("```mermaid\n")
        f.write(mermaid_syntax)
        f.write("\n```")
    print(f"Graph Mermaid visualization saved to: {mermaid_path}")

except Exception as e:
    print(f"\nCould not save graph visualization. This often means 'pygraphviz' is not installed correctly.")
    print(f"Please see installation instructions for your OS. Error details: {e}\n")
