import re
import json # Add json import if not already present from other functions
from . import file_io # Assuming file_io is in the same package

def sanitize_filename(title):
    """
    Cleans a string to be used as a valid filename.
    """
    if not title:
        return "untitled_paper"
    sanitized = re.sub(r'[\\/*?:"<>|]', "", title)
    sanitized = re.sub(r'[\s-]+', '-', sanitized)
    max_len = 150
    if len(sanitized) > max_len:
        # Ensure we don't cut in the middle of a multi-byte character if any
        # For simplicity, assuming standard characters; complex char handling might need more.
        if '-' in sanitized[:max_len]:
            sanitized = sanitized[:max_len].rsplit('-', 1)[0]
        else:
            sanitized = sanitized[:max_len]
    return sanitized.strip('-_')

def extract_paper_ids_from_search_data(search_data):
    """
    Extract paper IDs from loaded search results data.
    """
    if not search_data:
        return {}
        
    topic_paper_ids = {}
    for topic, topic_data in search_data.items():
        paper_ids = []
        if topic_data and 'data' in topic_data:
            for paper in topic_data['data']:
                paper_id = paper.get('paperId')
                if paper_id:
                    paper_ids.append(paper_id)
        topic_paper_ids[topic] = paper_ids
    return topic_paper_ids

def pool_paper_ids(topic_paper_ids):
    """
    Pools all paper IDs from different topics into a single list of unique IDs.
    """
    all_ids = set()
    for id_list in topic_paper_ids.values():
        all_ids.update(id_list)
    return list(all_ids)

def extract_pdf_url_from_paper_details(paper_details):
    """
    Extracts a downloadable PDF URL from paper details.
    """
    if not paper_details:
        return None

    open_access_pdf_info = paper_details.get('openAccessPdf')
    if open_access_pdf_info and open_access_pdf_info.get('url'):
        return open_access_pdf_info['url']

    if open_access_pdf_info and 'disclaimer' in open_access_pdf_info and open_access_pdf_info['disclaimer']:
        match = re.search(r'(https?://arxiv\.org/abs/[\w\.-]+)', open_access_pdf_info['disclaimer'])
        if match:
            arxiv_abs_url = match.group(1)
            return arxiv_abs_url.replace('/abs/', '/pdf/') + '.pdf'
    return None

def clean_recommendations_data(recommendations):
    """
    Prepares a flat list of recommendations with selected metadata.
    """
    clean_recommendations_list = []
    if not recommendations:
        return clean_recommendations_list

    for paper in recommendations:
        clean_paper = {
            'title': paper.get('title', 'No title'),
            'url': paper.get('url', ''), 
            'citationCount': paper.get('citationCount', 0),
            'externalIds': paper.get('externalIds', {}),
            'paperId': paper.get('paperId', 'Unknown'),
            'publicationDate': paper.get('publicationDate'),
            'references': paper.get('references', []), 
            'citations': paper.get('citations', []),
            'isOpenAccess': paper.get('isOpenAccess', False),
            'openAccessPdf': paper.get('openAccessPdf'), 
            'derivedPdfUrl': paper.get('derivedPdfUrl') 
        }
        clean_recommendations_list.append(clean_paper)
    return clean_recommendations_list

def extract_topics_from_json(file_path):
    """
    Extract topics from a JSON file containing query decomposition data.
    
    Args:
        file_path (str): Path to the JSON file
    
    Returns:
        list: List of topics extracted from the JSON, or an empty list on error.
    """
    data = file_io.load_json_file(file_path) # Use existing loader
    if not data:
        return [] # load_json_file already prints errors

    try:
        topics = data.get('analysis', {}).get('topics', [])
        return topics
    except Exception as e:
        print(f"Error extracting topics from loaded data: {e}")
        return []