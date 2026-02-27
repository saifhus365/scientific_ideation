import requests
import json
import time
import re
import os

# Ensure imports are relative to the current package structure if KG_explore is a package
# If running as a script directly from KG_explore, 'from .modules import ...' is correct.
# If KG_explore is not treated as a package by Python's path, you might need to adjust sys.path or run as 'python -m src.KG_explore.get_recommended_for_dblp'
from .modules import file_io
from .modules import data_processing
from .modules import semantic_scholar_api
from .modules import reporting


# Main execution
if __name__ == "__main__":
    # Configuration
    SEARCH_RESULTS_FILE = "results/paper_data/semantic_scholar_results1.json"
    # API_KEY can be set to None to use the default in semantic_scholar_api.py,
    # or provide a specific key here.
    API_KEY = '9ZRVL8BMOc8BxZVDMtOJm80LtqvGZWGCaQYmZr4X' 
    
    RECOMMENDATIONS_LIMIT = 15
    OUTPUT_FILE_RECOMMENDATIONS = "results/paper_data/recommended_papers_pooled1.json"
    DOWNLOAD_DIRECTORY = "data/papers/20250611_113410" 
    API_REQUEST_DELAY_S = 2 # Delay between fetching details for each paper

    # --- 1. Setup Download Directory ---
    try:
        os.makedirs(DOWNLOAD_DIRECTORY, exist_ok=True)
        print(f"Download directory '{DOWNLOAD_DIRECTORY}' is ready.")
    except OSError as e:
        print(f"Error creating directory {DOWNLOAD_DIRECTORY}: {e}")
        exit(1)

    # --- 2. Load Seed Paper IDs ---
    print("\nLoading and extracting paper IDs from search results...")
    search_data = file_io.load_json_file(SEARCH_RESULTS_FILE)
    
    if not search_data:
        print(f"Could not load search data from '{SEARCH_RESULTS_FILE}'. Exiting.")
        exit(1)
        
    topic_paper_ids = data_processing.extract_paper_ids_from_search_data(search_data)
    
    if not topic_paper_ids or not any(topic_paper_ids.values()):
        print("No paper IDs found in the search data. Please check your search results file.")
        exit(1)
        
    print("\nPooling seed papers from all topics...")
    all_seed_ids = data_processing.pool_paper_ids(topic_paper_ids)
    
    if not all_seed_ids:
        print("Could not find any seed paper IDs to use. Exiting.")
        exit(1)
    print(f"Found {len(all_seed_ids)} unique seed paper IDs.")

    # --- 3. Get Recommendations ---
    print(f"\nGetting {RECOMMENDATIONS_LIMIT} recommendations based on pooled seeds...")
    recommended_papers = semantic_scholar_api.get_recommendations(
        all_seed_ids, 
        api_key=API_KEY, 
        limit=RECOMMENDATIONS_LIMIT
    )
    
    # --- 4. Fetch Details & Download PDFs ---
    if recommended_papers:
        semantic_scholar_api.fetch_batch_paper_details(
            recommended_papers, 
            api_key=API_KEY,
            delay_seconds=API_REQUEST_DELAY_S
        )

        print("\nAttempting to download PDFs for recommended papers...")
        for paper in recommended_papers:
            title = paper.get('title')
            paper_id = paper.get('paperId')
            
            pdf_url = data_processing.extract_pdf_url_from_paper_details(paper)
            paper['derivedPdfUrl'] = pdf_url # Store for record-keeping

            if pdf_url:
                print(f"  Processing paper for download: '{title}' (ID: {paper_id})")
                file_io.download_pdf(
                    title=title,
                    paper_id=paper_id,
                    pdf_url=pdf_url,
                    download_dir=DOWNLOAD_DIRECTORY,
                    sanitize_func=data_processing.sanitize_filename
                )
            else:
                status = "Open Access" if paper.get('isOpenAccess') else "Not Open Access or Link not Found"
                print(f"  No downloadable PDF URL found for paper: '{title}' (ID: {paper_id}). Status: {status}")
    else:
        print("\nNo recommendations were generated. Skipping detail fetching and PDF download.")

    # --- 5. Process and Save Results ---
    if recommended_papers:
        cleaned_output_data = data_processing.clean_recommendations_data(recommended_papers)
        file_io.save_data_to_json(cleaned_output_data, OUTPUT_FILE_RECOMMENDATIONS)
        
        reporting.print_recommendations_summary(
            recommended_papers, 
            DOWNLOAD_DIRECTORY,
            data_processing.sanitize_filename
        )
        print(f"\nRecommendations processing completed! Check '{OUTPUT_FILE_RECOMMENDATIONS}' for results and '{DOWNLOAD_DIRECTORY}' for downloaded papers.")
    else:
        print("\nProcess finished, but no recommendations were generated or processed.")