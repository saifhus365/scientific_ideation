# A Persona-Based Collaborative Multi-Agent Framework for Scientific Idea Generation    

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation of the master thesis titled "A Persona-Based Collaborative Multi-Agent Framework for Scientific Idea Generation", a multi-agent system for automated scientific idea generation. The system takes a high-level research query, conducts an autonomous literature review, and generates novel research hypotheses through an agentic debate workflow.

---

## Table of Contents

1.  [Overview](#overview)
2.  [Key Features](#key-features)
3.  [Architecture](#architecture)
4.  [Repository Structure](#repository-structure)
5.  [Getting Started](#getting-started)
    *   [Prerequisites](#prerequisites)
    *   [Installation](#installation)
    *   [Configuration](#configuration)
6.  [Usage](#usage)
    *   [Running the Interactive Application](#running-the-interactive-application)
    *   [Running Experiments from Scripts](#running-experiments-from-scripts)
7.  [Reproducing Experiments](#reproducing-experiments)
    *   [Reproducing Main Results (Full Pipeline)](#reproducing-main-results-full-pipeline)
    *   [Reproducing Ablation Studies](#reproducing-ablation-studies)
    *   [Reproducing Zero-Shot Baseline](#reproducing-zero-shot-baseline)
8.  [Pipeline Stages Explained](#pipeline-stages-explained)
9.  [Evaluation Metrics](#evaluation-metrics)
10. [Results & Artifacts](#results--artifacts)
11. [Extending the Framework](#extending-the-framework)
12. [Citation](#citation)
13. [License](#license)

---

## Overview

Generating novel scientific ideas is a complex cognitive task. This project explores whether Large Language Models (LLMs), when organized in a multi-agent debate framework, can generate research hypotheses that are both novel and grounded in existing literature.

**VirSci** automates the research pipeline end-to-end:
1.  **Understands** a user's research query.
2.  **Discovers** relevant academic papers via Semantic Scholar.
3.  **Processes** and indexes the literature into a searchable knowledge base.
4.  **Generates** novel ideas through a multi-agent debate (simulating different scientific viewpoints).
5.  **Evaluates** the generated ideas for quality and novelty.

---

## Key Features

-   **Query Decomposition:** Analyzes natural language queries to extract key topics and user intent (e.g., exploratory, comparative).
-   **Agentic Literature Review:** An LLM-agent iteratively searches Semantic Scholar, scores papers for relevance, and builds a comprehensive literature bank.
-   **Automated PDF Processing:** Downloads and parses full-text PDFs to extract content beyond abstracts.
-   **Vector-Based Indexing (RAG):** Chunks and indexes paper content into ChromaDB, enabling Retrieval-Augmented Generation (RAG) for grounding idea generation.
-   **Multi-Agent Debate (LangGraph):** A state-machine orchestrates:
    *   **Persona Generation:** Creates a pool of diverse "virtual scientist" personas.
    *   **Team Selection:** Selects a team of debaters from the persona pool.
    *   **Idea Generation Rounds:** Each scientist proposes ideas, informed by RAG queries to the knowledge base.
    *   **Critic Rounds:** A "critic" agent challenges and evaluates the proposed ideas.
    *   **Summarization:** A "controller" summarizes each round for reflection.
    *   **Final Synthesis:** A "synthesizer" agent distills the debate history into a final list of novel hypotheses.
-   **Semantic Deduplication:** Uses sentence embeddings to filter out semantically similar ideas, ensuring a unique final output.
-   **Tournament-Based Evaluation:** An LLM-as-a-judge performs pairwise comparisons to rank ideas, calculating Precision@N metrics.
-   **Novelty Metrics:** Computes Historical Dissimilarity, Contemporary Dissimilarity, Contemporary Impact, and Overall Novelty against the discovered literature.
-   **Interactive Frontend:** A Streamlit web application provides a user-friendly interface for running and visualizing the workflow.
-   **Ablation Studies:** Configurable flags allow for systematic ablation of key components (RAG, Critique, Viewpoints, Synthesis).

---

## Architecture

The system architecture consists of a modular, stage-based pipeline controlled by a central orchestrator.

### End-to-End Application Flow

The following diagram illustrates the complete pipeline, from user input to the final ranked ideas.

### Agentic Workflow Design (Debate Stage)

This diagram details the internal state machine of the multi-agent idea generation system built with LangGraph.



---

## Repository Structure

```
master_thesis_saif/
│
├── src/                            # Main source code directory
│   ├── app.py                      # Streamlit web application entry point
│   ├── main.py                     # Core pipeline stage definitions
│   ├── pipeline_runner.py          # Single-run pipeline orchestrator for experiments
│   ├── test_queries.json           # Test queries for batch experiments
│   │
│   ├── query_decomp/               # Stage 1: Query Analysis
│   │   ├── config.py               # LLM configuration
│   │   ├── llm_client.py           # LLM API client wrapper
│   │   ├── query_analyzer.py       # Main query analysis logic
│   │   └── response_parser.py      # Pydantic models for parsed query
│   │
│   ├── literature_review/          # Stage 2: Agentic Literature Review
│   │   ├── agent.py                # The LitReviewAgent class
│   │   ├── s2_search.py            # Semantic Scholar API wrapper
│   │   └── tools.py                # Helper functions (search, filter, format)
│   │
│   ├── paper_processing/           # Stage 3 & 4: PDF Download & Parsing
│   │   └── processor.py            # PDF download and text extraction logic
│   │
│   ├── data_indexing/              # Stage 5: Vector Database Indexing
│   │   └── indexer.py              # ChromaDB indexing logic
│   │
│   ├── agentic_workflow/           # Stage 6: Multi-Agent Idea Generation
│   │   ├── graph.py                # LangGraph state machine definition
│   │   ├── state.py                # AgentState TypedDict
│   │   ├── data_models.py          # Pydantic models for debate components
│   │   ├── agent_builders.py       # Functions to create agents (scientists, critic)
│   │   ├── prompts.json            # All LLM prompts for the multi-agent debate
│   │   └── run.py                  # Standalone script to run the debate workflow
│   │
│   ├── experiments/                # Baseline experiments (Zero-shot, simplified)
│   │   ├── simple_run.py           # Simplified agent workflow for baseline
│   │   ├── simple_run_generate_only.py # Zero-shot generation baseline
│   │   └── zeroshot_graph.py       # LangGraph for zero-shot baseline
│   │
│   ├── metrics/                    # Stage 7 & 8: Evaluation and Metrics
│   │   ├── deduplication.py        # Cosine similarity-based deduplication
│   │   ├── llm_evaluation.py       # LLM-as-a-judge tournament ranking
│   │   ├── get_precisions.py       # Precision@N calculation
│   │   ├── novelty.py              # Novelty metrics (HD, CD, CI, ON)
│   │   ├── evaluation_runner.py    # Orchestrates all evaluation steps
│   │   ├── generate_figures.py     # Scripts to generate publication figures
│   │   ├── statistical_analysis.py # Statistical significance tests
│   │   └── significance_test_VirSci.py # Comparison with VirSci baseline
│   │
│   └── results/                    # Run-specific outputs (within src)
│       ├── agent_states/           # Saved final states of agentic workflows
│       └── final_reports/          # JSON reports of discovered papers
│
├── results/                        # Project-wide aggregated results
│   ├── agent_states/               # All workflow states from full pipeline
│   ├── experiment_summaries/       # Aggregated metrics from batch experiments
│   ├── evaluation_summary/         # Detailed evaluation outputs
│   ├── graph_visuals/              # Pipeline diagrams and visualizations
│   ├── simple_agent_states/        # States from simplified baseline runs
│   ├── zeroshot_agent_states/      # States from zero-shot baseline runs
│   └── VirSci_results/             # Results compared against VirSci baseline
│
├── VirSci-eval/                    # Code for VirSci comparison experiments
│
├── docs/                           # Documentation (papers, presentations)
│   ├── papers/
│   └── presentations/
│
├── run_experiments.py              # Main script to run full ablation experiments
├── run_zeroshot_experiment.py      # Script to run zero-shot baseline experiments
├── requirements.txt                # Python dependencies
├── .gitignore
└── README.md                       # This file
```

---

## Getting Started

### Prerequisites

-   **Python 3.9+**
-   **pip** (Python package manager)
-   **Git**
-   **API Keys:**
    -   **Mistral AI API Key:** Required for all LLM-based components.
    -   **Semantic Scholar API Key:** Required for literature review (optional, but recommended for higher rate limits).

### Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd master_thesis_saif
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    # On macOS/Linux
    python -m venv venv
    source venv/bin/activate

    # On Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK Data (for sentence tokenization):**
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('punkt_tab')
    ```

### Configuration

Create a `.env` file in the project root with your API keys:

```env
# Required
MISTRAL_API_KEY="your_mistral_api_key_here"

# Recommended (for higher Semantic Scholar rate limits)
S2_API_KEY="your_semantic_scholar_api_key"
```

**How to obtain API keys:**
-   **Mistral AI:** Sign up at [https://console.mistral.ai/](https://console.mistral.ai/)
-   **Semantic Scholar:** Request an API key at [https://www.semanticscholar.org/product/api](https://www.semanticscholar.org/product/api)

---

## Usage

### Running the Interactive Application

The Streamlit application is the easiest way to explore the system.

1.  Navigate to the `src` directory:
    ```bash
    cd src
    ```

2.  Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

3.  Open your browser to the displayed URL (usually `http://localhost:8501`).

4.  Enter a research query (e.g., "Explore how microbial dark matter modeling could enhance bioinformatics pipelines") and click **"Start Full Research Workflow"**.

5.  The application will display progress for each stage and present the final generated ideas.



---

## Reproducing Experiments

This section provides step-by-step instructions to reproduce the experiments from the thesis.

### Reproducing Main Results (Full Pipeline)

The main experiments run the full pipeline on a set of 50 test queries across different ablation configurations.

1.  **Ensure Configuration:** Verify that your `.env` file contains valid API keys.

2.  **Run the Main Experiment Script:**
    ```bash
    python run_experiments.py
    ```

3.  **Monitor Progress:** The script will print progress for each query and configuration. Results are saved incrementally.

4.  **Output Location:**
    -   **Individual Run States:** `results/agent_states/workflow_state_<timestamp>.json`
    -   **Aggregated Summary:** `results/experiment_summaries/live_summary_<config_name>.json`

5.  **Estimated Time:** Each query takes approximately 10-15 minutes. A full run of 50 queries × multiple configurations can take several days.

### Reproducing Ablation Studies

The `run_experiments.py` script supports multiple ablation configurations. To run specific ablations:

> **Note:** The **Full_System** baseline does **NOT** include the synthesizer agent. The `With_Synthesizer` ablation **adds** the synthesizer agent to study its effect.

1.  **Edit `run_experiments.py`:** Uncomment the desired configurations in `ABLATION_CONFIGS`:
    ```python
    ABLATION_CONFIGS = [
        {
            "name": "Full_System",  # Baseline: without synthesizer
            "use_ablation_synthesis": True, "use_ablation_RAG": False,
            "use_ablation_viewpoint": False, "use_ablation_critique": False,
        },
        {
            "name": "With_Synthesizer",  # Adds synthesizer agent
            "use_ablation_synthesis": False, "use_ablation_RAG": False,
            "use_ablation_viewpoint": False, "use_ablation_critique": False,
        },
        {
            "name": "No_Critique",
            "use_ablation_synthesis": True, "use_ablation_RAG": False,
            "use_ablation_viewpoint": False, "use_ablation_critique": True,  # Ablated
        },
        {
            "name": "No_RAG",
            "use_ablation_synthesis": True, "use_ablation_RAG": True,  # Ablated
            "use_ablation_viewpoint": False, "use_ablation_critique": False,
        },
        # Add more as needed...
    ]
    ```

2.  **Run the script:**
    ```bash
    python run_experiments.py
    ```

**Available Ablation Flags:**
| Flag | Description |
|------|-------------|
| `use_ablation_synthesis` | When `True` (default): disables synthesizer agent. When `False`: enables synthesizer agent |
| `use_ablation_RAG` | Disables RAG-based context retrieval during idea generation |
| `use_ablation_critique` | Removes the critic agent rounds |
| `use_ablation_viewpoint` | Uses uniform scientist personas instead of diverse viewpoints |
| `use_ablation_query_decomp` | Skips query decomposition and uses raw query as topic |

### Reproducing Zero-Shot Baseline

The zero-shot baseline generates ideas without the multi-agent debate framework.

1.  **Run the Zero-Shot Experiment Script:**
    ```bash
    python run_zeroshot_experiment.py
    ```

2.  **Output Location:**
    -   **Individual Run States:** `results/zeroshot_agent_states/workflow_state_<timestamp>.json`
    -   **Aggregated Summary:** `results/experiment_summaries/live_summary_zeroshot.json`

---

## Pipeline Stages Explained

This section provides a detailed breakdown of each pipeline stage, including the LLM agents involved and links to the relevant code and prompts.

### Stage 1: Query Decomposition

**Purpose:** Analyzes the user's natural language research query to extract structured information.

| Component | Description | Code Location |
|-----------|-------------|---------------|
| **Query Analyzer** | Parses query to extract topics, timeline, and user intention | [`src/query_decomp/query_analyzer.py`](src/query_decomp/query_analyzer.py) |
| **LLM Prompt** | System prompt for query analysis | [`src/query_decomp/config.py`](src/query_decomp/config.py) (lines 23-50) |

**Outputs:** `QueryAnalysis` object containing:
- `topics`: 1-3 key research topics
- `timeline`: Date range (if applicable)
- `intention`: One of `Exploratory`, `Comparative`, `Descriptive`, `Causal`, or `Relational`

---

### Stage 2: Literature Review

**Purpose:** An agentic loop that iteratively searches Semantic Scholar, scores papers for relevance, and builds a comprehensive paper bank.

| Component | Description | Code Location |
|-----------|-------------|---------------|
| **LitReviewAgent** | Main agent orchestrating the review | [`src/literature_review/agent.py`](src/literature_review/agent.py) |
| **Strategy Selector Prompt** | LLM decides next search strategy (KeywordQuery, PaperQuery, GetReferences) | [`src/literature_review/agent.py`](src/literature_review/agent.py) (lines 27-49) |
| **Paper Scorer Prompt** | LLM scores papers (1-10) for relevance | [`src/literature_review/agent.py`](src/literature_review/agent.py) (lines 62-76) |
| **Semantic Scholar Tools** | API wrappers for paper search | [`src/literature_review/tools.py`](src/literature_review/tools.py) |

**Outputs:** A ranked list of discovered papers with relevance scores.

---

### Stage 3 & 4: Paper Downloading & Processing

**Purpose:** Downloads PDFs for discovered papers and extracts text content.

| Component | Description | Code Location |
|-----------|-------------|---------------|
| **PDF Downloader** | Downloads papers from open access sources | [`src/paper_processing/processor.py`](src/paper_processing/processor.py) |
| **PDF Parser** | Extracts structured text from PDFs | [`src/paper_processing/processor.py`](src/paper_processing/processor.py) |

**Outputs:** JSON files containing parsed paper content organized by section.

---

### Stage 5: Data Indexing (RAG Setup)

**Purpose:** Chunks paper text and indexes into ChromaDB for retrieval-augmented generation.

| Component | Description | Code Location |
|-----------|-------------|---------------|
| **Indexer** | Chunks text and creates embeddings | [`src/data_indexing/indexer.py`](src/data_indexing/indexer.py) |
| **Text Splitter** | Recursive text chunking (10,000 chars, 2,000 overlap) | [`src/data_indexing/indexer.py`](src/data_indexing/indexer.py) (lines 12-18) |

**Outputs:** A ChromaDB collection named `lit_review_papers_<timestamp>`.

---

### Stage 6: Agentic Idea Generation (Multi-Agent Debate)

**Purpose:** A multi-round debate among virtual scientist personas to generate novel research ideas.

This is the core innovation of the framework. The debate is orchestrated by a LangGraph state machine.

| LLM Agent | Role | Prompt Location | Code Location |
|-----------|------|-----------------|---------------|
| **Persona Pool Generator** | Creates 5-10 diverse scientist personas based on the query | [`prompts.json → persona_pool_prompt`](src/agentic_workflow/prompts.json) | [`graph.py → generate_persona_pool()`](src/agentic_workflow/graph.py#L25-L37) |
| **Team Selector** | Selects 3 personas to form the debate team | [`prompts.json → team_selection_prompt`](src/agentic_workflow/prompts.json) | [`graph.py → select_team_from_pool()`](src/agentic_workflow/graph.py#L39-L66) |
| **Idea Generator (Scientist)** | Each scientist proposes 3-5 ideas using RAG context | [`prompts.json → idea_generation_agent_prompt`](src/agentic_workflow/prompts.json) | [`graph.py → idea_generation_round()`](src/agentic_workflow/graph.py#L68-L134) |
| **Critic** | Evaluates ideas for novelty, feasibility, and impact | [`prompts.json → critic_agent_prompt`](src/agentic_workflow/prompts.json) | [`graph.py → critic_round()`](src/agentic_workflow/graph.py#L137-L151) |
| **Moderator (Summarizer)** | Creates a summary of each round for reflection | [`prompts.json → round_summary_prompt`](src/agentic_workflow/prompts.json) | [`graph.py → summarize_round()`](src/agentic_workflow/graph.py#L154-L193) |
| **Synthesizer** | Distills debate history into final 20 novel ideas | [`prompts.json → final_synthesis_prompt`](src/agentic_workflow/prompts.json) | [`graph.py → synthesize_final_ideas()`](src/agentic_workflow/graph.py#L227-L261) |
| **Abstract Generator** | Generates scientific abstracts for final ideas | [`prompts.json → abstract_generation_prompt`](src/agentic_workflow/prompts.json) | [`graph.py → generate_abstracts()`](src/agentic_workflow/graph.py#L195-L224) |

**Debate Flow:**
1. **Persona Pool Generation** → **Team Selection**
2. **For each round (3 rounds):**
   - Scientists generate ideas (with RAG context)
   - Critic evaluates ideas
   - Moderator summarizes round
3. **Final Synthesis** → **Abstract Generation**

**Outputs:** `FinalIdeaList` containing 20 novel research hypotheses with abstracts.

---



### Stage 7: Evaluation & Ranking

**Purpose:** Evaluates and ranks the generated ideas using LLM-as-a-judge and computes quality metrics.

| Component | Description | Prompt Location | Code Location |
|-----------|-------------|-----------------|---------------|
| **Tournament Ranker** | Pairwise comparison of ideas | Inline prompt in [`llm_evaluation.py`](src/metrics/llm_evaluation.py) (lines 13-22) | [`src/metrics/llm_evaluation.py`](src/metrics/llm_evaluation.py) |
| **Precision Calculator** | Computes Precision@N | N/A | [`src/metrics/get_precisions.py`](src/metrics/get_precisions.py) |
| **Novelty Calculator** | Computes HD, CD, CI, ON metrics | N/A | [`src/metrics/novelty.py`](src/metrics/novelty.py) |

---

## LLM Prompts Reference

All prompts used in the multi-agent debate are stored in a centralized location for easy modification:

**📁 Location:** [`src/agentic_workflow/prompts.json`](src/agentic_workflow/prompts.json)

| Prompt Key | Agent | Purpose |
|------------|-------|---------|
| `persona_pool_prompt` | Persona Pool Generator | Generate diverse scientist personas |
| `team_selection_prompt` | Team Selector | Select optimal debate team |
| `idea_generation_agent_prompt` | Scientist (Idea Generator) | Generate research ideas with RAG |
| `ablation_idea_generation_agent_prompt` | Scientist (Ablation) | Generate more ideas (17-20) for ablation |
| `critic_agent_prompt` | Critic | Evaluate ideas for novelty/feasibility/impact |
| `round_summary_prompt` | Moderator | Summarize round with critic feedback |
| `ablation_round_summary_prompt` | Moderator (Ablation) | Summarize round without critic feedback |
| `final_synthesis_prompt` | Synthesizer | Synthesize debate history into final ideas |
| `ablation_final_synthesis_prompt` | Synthesizer (Ablation) | Synthesize only final round |
| `abstract_generation_prompt` | Abstract Generator | Generate scientific abstracts |

**Other Prompt Locations:**

| Module | Prompt Location |
|--------|-----------------|
| Query Decomposition | [`src/query_decomp/config.py`](src/query_decomp/config.py) → `PromptTemplates.query_analysis` |
| Literature Review (Strategy) | [`src/literature_review/agent.py`](src/literature_review/agent.py) → `_get_next_query_from_llm()` |
| Literature Review (Scoring) | [`src/literature_review/agent.py`](src/literature_review/agent.py) → `_score_papers_with_llm()` |
| Tournament Ranking | [`src/metrics/llm_evaluation.py`](src/metrics/llm_evaluation.py) → `better_idea()`

---

## Evaluation Metrics

### Precision@N
Ideas are ranked using a tournament-style pairwise comparison by an LLM judge. Precision@N measures the proportion of "non-baseline" (full system) ideas in the top N positions.

### Novelty Metrics
-   **Historical Dissimilarity (HD):** Euclidean distance to pre-2023 papers in embedding space.
-   **Contemporary Dissimilarity (CD):** Euclidean distance to contemporary (2023+) papers.
-   **Contemporary Impact (CI):** Average citation count of similar contemporary papers.
-   **Overall Novelty (ON):** Composite score: `(HD × CI) / CD`

---

## Results & Artifacts

After running experiments, the following artifacts are generated:

| Artifact | Location | Description |
|----------|----------|-------------|
| Workflow States | `results/agent_states/` | Full state snapshots from each run |
| Literature Reports | `results/final_reports/` | Lists of discovered papers per run |
| Experiment Summaries | `results/experiment_summaries/` | Aggregated precision and novelty metrics |
| Graph Visualizations | `results/graph_visuals/` | Pipeline and agent workflow diagrams |
| ChromaDB Data | `chroma_db/` | Persistent vector database (created at runtime) |

### Generating Figures

To regenerate publication-quality figures from experiment results:

```bash
cd src/metrics
python generate_figures.py
```

Figures will be saved to the project root (e.g., `figure1_precision_bars.png`).

---

## Extending the Framework

### Adding New Test Queries
Edit `src/test_queries.json`:
```json
{
  "queries": [
    "Your new research query here",
    ...
  ]
}
```

### Adding New Ablation Configurations
Modify `ABLATION_CONFIGS` in `run_experiments.py` with new flag combinations.

### Customizing LLM Prompts
All prompts are centralized in `src/agentic_workflow/prompts.json`. Modify these to adjust agent behavior.

### Switching LLM Providers
The system uses LangChain, making it provider-agnostic. To switch:
1.  Install the desired LangChain integration (e.g., `langchain-openai`)
2.  Update imports in `src/agentic_workflow/graph.py` and other files
3.  Update the LLM initialization with appropriate API keys

---

## Citation

If you use this code in your research, please cite:

```bibtex
@MastersThesis{saif:2026,
  author =                   {Husain Qasim Ali Saif},
  month =                    jan,
  school =                   {Bauhaus-Universit{\"a}t Weimar, Fakult{\"a}t Medien, Computer Science for Digital Media},
  title =                    {{A Persona-Based Collaborative Multi-Agent Framework for Scientific Idea Generation}},
  type =                     {Masterarbeit},
  year =                     2026
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

-   [Semantic Scholar API](https://www.semanticscholar.org/product/api) for academic paper search
-   [LangGraph](https://github.com/langchain-ai/langgraph) for multi-agent orchestration
-   [ChromaDB](https://www.trychroma.com/) for vector storage
-   [Mistral AI](https://mistral.ai/) for LLM inference
