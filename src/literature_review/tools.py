import requests
import re
import json
from pathlib import Path
import os
from dotenv import load_dotenv
import time

# --- Configuration ---
S2_API_URL = 'https://api.semanticscholar.org/graph/v1'
REC_API_URL = "https://api.semanticscholar.org/recommendations/v1/papers/forpaper/"
S2_API_KEY = None  # Placeholder, will be loaded dynamically

def initialize_api_key():
    """Loads the Semantic Scholar API key from environment variables."""
    global S2_API_KEY
    if S2_API_KEY: # Don't reload if it's already loaded
        return

    # Load environment variables from a .env file at the project root
    load_dotenv()
    
    S2_API_KEY = os.getenv("S2_API_KEY")
    if not S2_API_KEY:
        print("Warning: 'S2_API_KEY' not found in your environment variables or .env file.")
        print("         API calls to Semantic Scholar may fail.")

# --- Core API Functions ---

def search_papers_by_keyword(keyword: str, limit: int = 20) -> list:
    """Retrieves papers from Semantic Scholar based on a keyword search."""
    if not S2_API_KEY: initialize_api_key()
    
    time.sleep(3)
    search_url = f"{S2_API_URL}/paper/search"
    query_params = {
        'query': keyword,
        'limit': limit,
        'fields': 'title,year,citationCount,abstract,tldr,authors,venue'
    }
    headers = {'x-api-key': S2_API_KEY}
    
    try:
        response = requests.get(search_url, params=query_params, headers=headers)
        response.raise_for_status()
        return response.json().get("data", [])
    except requests.exceptions.RequestException as e:
        print(f"Error in keyword search for '{keyword}': {e}")
        return []

def get_recommendations_for_paper(paper_id: str, limit: int = 20) -> list:
    """Retrieves recommended papers based on a given paper ID."""
    if not S2_API_KEY: initialize_api_key()

    rec_url = f"{REC_API_URL}{paper_id}"
    query_params = {
        'limit': limit,
        'fields': 'title,year,citationCount,abstract,authors,venue'
    }
    headers = {'x-api-key': S2_API_KEY}

    try:
        response = requests.get(rec_url, params=query_params, headers=headers)
        response.raise_for_status()
        return response.json().get("recommendedPapers", [])
    except requests.exceptions.RequestException as e:
        print(f"Error getting recommendations for paper '{paper_id}': {e}")
        return []

def get_paper_details(paper_id: str, fields: str = 'title,year,abstract,authors,citationCount,venue,citations,references,tldr') -> dict:
    """Gets detailed information for a single paper."""
    if not S2_API_KEY: initialize_api_key()

    detail_url = f"{S2_API_URL}/paper/{paper_id}"
    query_params = {'fields': fields}
    headers = {'x-api-key': S2_API_KEY}

    time.sleep(3)
    try:
        response = requests.get(detail_url, params=query_params, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting details for paper '{paper_id}': {e}")
        return {}

def get_references(paper_id: str) -> list:
    """Gets the list of papers referenced by a given paper."""
    paper_details = get_paper_details(paper_id, fields='references.paperId,references.title,references.year')
    return paper_details.get("references", [])

# --- Utility and Formatting Functions ---

def filter_papers(paper_list: list) -> list:
    """Filters out survey, review, or position papers."""
    filtered_list = []
    for paper in paper_list:
        title = paper.get("title", "").lower()
        if "survey" in title or "review" in title or "position paper" in title:
            continue
        # Ensure the paper has a valid ID and an abstract
        if paper.get("paperId") and paper.get("abstract"):
             filtered_list.append(paper)
    return filtered_list

def format_papers_for_llm(paper_list: list, include_abstract=True) -> str:
    """Converts a list of papers to a string for use in an LLM prompt."""
    output_str = ""
    for paper in paper_list:
        output_str += f"paperId: {paper.get('paperId', 'N/A')}\n"
        output_str += f"title: {paper.get('title', 'N/A').strip()}\n"
        
        if include_abstract:
            abstract = paper.get("abstract", "")
            tldr = paper.get("tldr", {}).get("text") if paper.get("tldr") else None
            
            if abstract:
                output_str += f"abstract: {abstract.strip()}\n"
            elif tldr:
                output_str += f"tldr: {tldr.strip()}\n"
        
        output_str += "\n"
    return output_str

def dedup_paper_list(paper_list: list) -> list:
    """Deduplicates a list of papers based on title and abstract."""
    seen_titles = set()
    seen_abstracts = set()
    deduped_list = []
    
    for paper in paper_list:
        title = ''.join(paper.get('title', '').lower().split())
        abstract = paper.get('abstract', '')
        
        if title in seen_titles or (abstract and abstract in seen_abstracts):
            continue
            
        seen_titles.add(title)
        if abstract:
            seen_abstracts.add(abstract)
        deduped_list.append(paper)
        
    return deduped_list 