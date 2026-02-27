import requests
import time

DEFAULT_API_KEY = '9ZRVL8BMOc8BxZVDMtOJm80LtqvGZWGCaQYmZr4X' 

def get_recommendations(paper_ids, api_key=None, limit=20):
    """
    Get recommended papers from Semantic Scholar API.
    """
    if not paper_ids:
        print("No paper IDs provided for recommendations.")
        return []
    
    url = "https://api.semanticscholar.org/recommendations/v1/papers"
    query_params = {
        "fields": "title,url,citationCount,externalIds,paperId,publicationDate",
        "limit": str(limit)
    }
    payload = {"positivePaperIds": paper_ids}
    
    headers = {}
    key_to_use = api_key if api_key else DEFAULT_API_KEY
    if key_to_use:
        headers["x-api-key"] = key_to_use
    
    try:
        print(f"Requesting {limit} recommendations based on {len(paper_ids)} seed paper IDs...")
        response = requests.post(url, params=query_params, json=payload, headers=headers)
        response.raise_for_status()
        
        api_response = response.json()
        if api_response and 'recommendedPapers' in api_response:
            papers = api_response['recommendedPapers']
            papers.sort(key=lambda p: p.get("citationCount", 0), reverse=True)
            print(f"Received {len(papers)} recommended papers from API.")
            return papers
        else:
            print("No recommendations found in the API response.")
            return []
            
    except requests.exceptions.RequestException as e:
        print(f"Error getting recommendations: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.text}")
        return []

def fetch_batch_paper_details(papers_list, api_key=None, delay_seconds=1):
    """
    Fetches detailed information for a list of papers, updating them in-place.
    """
    if not papers_list:
        return []
        
    print(f"\nFetching details for {len(papers_list)} recommended papers...")
    headers = {}
    key_to_use = api_key if api_key else DEFAULT_API_KEY
    if key_to_use:
        headers["x-api-key"] = key_to_use
        
    for i, paper in enumerate(papers_list):
        paper_id = paper.get('paperId')
        title = paper.get('title', 'Unknown Title')
        if not paper_id:
            print(f"  Skipping paper at index {i} due to missing paperId.")
            continue

        if i > 0: time.sleep(delay_seconds)
        
        print(f"  Fetching details for paper {i+1}/{len(papers_list)}: '{title}' (ID: {paper_id})...")
        
        detail_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
        params = {"fields": "references.paperId,citations.paperId,openAccessPdf,isOpenAccess,publicationDate"}
        
        try:
            response = requests.get(detail_url, params=params, headers=headers)
            response.raise_for_status()
            details = response.json()
            
            paper['references'] = details.get('references', [])
            paper['citations'] = details.get('citations', [])
            paper['openAccessPdf'] = details.get('openAccessPdf')
            paper['isOpenAccess'] = details.get('isOpenAccess', False)
            if not paper.get('publicationDate') and details.get('publicationDate'):
                 paper['publicationDate'] = details.get('publicationDate')

        except requests.exceptions.RequestException as e:
            print(f"    Could not fetch details for paper '{title}' (ID: {paper_id}): {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"    Response content: {e.response.text}")
            paper.setdefault('references', [])
            paper.setdefault('citations', [])
            paper.setdefault('openAccessPdf', None)
            paper.setdefault('isOpenAccess', False)
            
    return papers_list

def search_papers_by_keyword(query, api_key=None, year_filter="2020-", limit=10):
    """
    Search Semantic Scholar API for papers related to a specific keyword query.
    
    Args:
        query (str): Search query.
        api_key (str, optional): Semantic Scholar API key. Defaults to DEFAULT_API_KEY.
        year_filter (str, optional): Year filter (e.g., "2020-" or "2020-2022"). Defaults to "2020-".
        limit (int, optional): Maximum number of papers to return. Defaults to 10.
    
    Returns:
        dict: API response containing paper data, or None on error.
    """
    url = "https://api.semanticscholar.org/graph/v1/paper/search" # Corrected endpoint
    
    query_params = {
        "query": query, # No need for extra quotes if the API handles it, or add if necessary based on API behavior
        "fields": "title,paperId,publicationDate,citationCount", # Added citationCount and publicationDate
        "year": year_filter,
        "limit": limit
    }
    
    headers = {}
    key_to_use = api_key if api_key else DEFAULT_API_KEY
    if key_to_use:
        headers["x-api-key"] = key_to_use
    
    try:
        response = requests.get(url, params=query_params, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error searching Semantic Scholar for query '{query}': {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.text}")
        return None

def search_papers_for_topics_bulk(topics, api_key=None, year_filter="2020-", delay_seconds=1, papers_per_topic=3):
    """
    Search multiple topics using Semantic Scholar API.
    
    Args:
        topics (list): List of topics to search.
        api_key (str, optional): Semantic Scholar API key. Defaults to DEFAULT_API_KEY.
        year_filter (str, optional): Year filter for publications. Defaults to "2020-".
        delay_seconds (int, optional): Delay between API calls in seconds. Defaults to 1.
        papers_per_topic (int, optional): Maximum number of papers to retrieve per topic. Defaults to 3.
    
    Returns:
        dict: Dictionary with topics as keys and search results (API response dict) as values.
    """
    results = {}
    key_to_use = api_key if api_key else DEFAULT_API_KEY
    
    for i, topic in enumerate(topics, 1):
        print(f"Searching topic {i}/{len(topics)}: {topic}")
        
        search_result = search_papers_by_keyword(
            query=topic, 
            api_key=key_to_use, 
            year_filter=year_filter, 
            limit=papers_per_topic
        )
        
        if search_result and 'data' in search_result:
            results[topic] = search_result # Store the full API response for the topic
            paper_count = len(search_result['data'])
            print(f"  Found {paper_count} papers for '{topic}' (requested {papers_per_topic})")
        else:
            results[topic] = None # Or an empty structure like {'data': [], 'total': 0}
            print(f"  No results found or error for '{topic}'")
        
        if i < len(topics) and delay_seconds > 0:
            time.sleep(delay_seconds)
            
    return results