import json
import time
from . import tools
from query_decomp.llm_client import LLMClient
from query_decomp.config import LLMConfig

# --- Constants ---
MAX_ITERATIONS = 5
PAPERS_PER_ITERATION = 10
GROUNDING_PAPERS_K = 5

class LitReviewAgent:
    def __init__(self, initial_query: str):
        self.initial_query = initial_query
        self.paper_bank = {}  # Using a dict for easy deduplication by paperId
        self.llm_client = LLMClient(LLMConfig())
        self.past_queries = []

    def _get_next_query_from_llm(self) -> str:
        """Asks the LLM to generate the next query based on the current paper bank."""
        
        # Sort papers by score to get top papers for grounding
        sorted_papers = sorted(self.paper_bank.values(), key=lambda p: p.get("score", 0), reverse=True)
        grounding_papers = sorted_papers[:GROUNDING_PAPERS_K]
        grounding_papers_str = tools.format_papers_for_llm(grounding_papers, include_abstract=False)

        prompt = f"""
        You are a research assistant doing a literature review. Your goal is to build a comprehensive list of relevant papers.
        
        You have access to the following functions:
        1. KeywordQuery("keyword"): Search for papers using a keyword. Good for broad exploration.
        2. PaperQuery("paperId"): Find papers similar to a given paper. Good for deepening a thread.
        3. GetReferences("paperId"): Get the papers cited by a given paper. Good for finding foundational work.

        You have already run the following queries:
        {self.past_queries}

        Based on the current top papers in your collection, generate a NEW, DIVERSE query to expand the search.
        Current Top Papers:
        ---
        {grounding_papers_str}
        ---

        Formulate your new query as a single function call (e.g., KeywordQuery("new deep learning methods")). 
        DO NOT provide any other text or explanation. 
        DO no write this is in a block as code. All i want is the plain query.
        DO NOT combine more than 3 search entities.
        Each concept should be nice and concise not more than 3 words.
        """

        response = self.llm_client.generate_response(prompt)
        # Simple parsing for the function call
        return response.strip()

    def _score_papers_with_llm(self, new_papers: list) -> dict:
        """Uses the LLM to score a list of papers based on the initial query."""
        if not new_papers:
            return {}
            
        papers_to_score_str = tools.format_papers_for_llm(new_papers)
        
        prompt = f"""
        You are a research assistant. Your task is to score papers for their relevance to the following research topic:
        "{self.initial_query}"

        Score each paper from 1 to 10 based on its direct relevance. A score of 10 means it is extremely relevant.
        Focus on papers that propose novel methods or findings. Give lower scores to surveys, reviews, or tangentially related work.

        Here are the papers to score:
        ---
        {papers_to_score_str}
        ---

        Provide your response as a single JSON object where keys are paperIds and values are the integer scores.
        Example: {{"paperId1": 8, "paperId2": 5}}
        """
        response_str = self.llm_client.generate_response(prompt)
        
        try:
            # Clean the response to grab only the JSON object
            json_str = response_str[response_str.find('{'):response_str.rfind('}')+1]
            return json.loads(json_str)
        except (json.JSONDecodeError, IndexError):
            print(f"Warning: Could not decode LLM scoring response: {response_str}")
            return {}

    def _execute_query(self, query: str) -> list:
        """Parses and executes a tool function call, ignoring LLM thoughts."""
        self.past_queries.append(query)  # Save raw response for history

        # The tool call is expected to be after the </think> block if it exists.
        cleaned_query = query
        if "</think>" in query:
            # Take the part after the last </think> tag and strip whitespace
            cleaned_query = query.rsplit("</think>", 1)[-1].strip()

        print(f"Executing query: {cleaned_query}")
        
        try:
            if cleaned_query.startswith("KeywordQuery"):
                keyword = cleaned_query[len("KeywordQuery(\""):-2]
                return tools.search_papers_by_keyword(keyword, limit=PAPERS_PER_ITERATION)
            elif cleaned_query.startswith("PaperQuery"):
                paper_id = cleaned_query[len("PaperQuery(\""):-2]
                return tools.get_recommendations_for_paper(paper_id, limit=PAPERS_PER_ITERATION)
            elif cleaned_query.startswith("GetReferences"):
                paper_id = cleaned_query[len("GetReferences(\""):-2]
                return tools.get_references(paper_id)
            else:
                print(f"Warning: Unknown query format: {cleaned_query}")
                return []
        except Exception as e:
            print(f"Error executing query '{cleaned_query}': {e}")
            return []

    def run(self):
        """Runs the iterative literature review process."""
        print("--- Starting Agentic Literature Review ---")

        # Initial seed query
        current_query = f"KeywordQuery(\"{self.initial_query}\")"
        
        for i in range(MAX_ITERATIONS):
            print(f"\n--- Iteration {i+1}/{MAX_ITERATIONS} ---")
            
            new_papers = self._execute_query(current_query)
            
            if not new_papers:
                print("No new papers found for this query.")
                # If a query fails, get a new one from the LLM
                current_query = self._get_next_query_from_llm()
                continue

            # Filter and score new papers
            new_papers = tools.filter_papers(new_papers)
            unseen_papers = [p for p in new_papers if p['paperId'] not in self.paper_bank]
            
            if not unseen_papers:
                print("All papers found are already in the bank.")
                current_query = self._get_next_query_from_llm()
                continue
                
            print(f"Found {len(unseen_papers)} new, relevant papers to score.")
            scores = self._score_papers_with_llm(unseen_papers)

            # Add scored papers to the bank
            for paper in unseen_papers:
                paper_id = paper['paperId']
                paper['score'] = scores.get(paper_id, 0)
                self.paper_bank[paper_id] = paper

            print(f"Paper bank now contains {len(self.paper_bank)} papers.")
            
            # Get the next query from the LLM
            current_query = self._get_next_query_from_llm()
            time.sleep(5) # Small delay to avoid API rate limits

        print("\n--- Literature Review Finished ---")
        # Return the final, sorted list of papers
        final_list = sorted(self.paper_bank.values(), key=lambda p: p.get("score", 0), reverse=True)
        return tools.dedup_paper_list(final_list) 