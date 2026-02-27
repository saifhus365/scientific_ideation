import os
import re # For more robust timestamp extraction
from .modules import data_processing
from .modules import semantic_scholar_api
from .modules import reporting
from .modules import file_io

# Main execution
if __name__ == "__main__":
    # --- Configuration ---
    # General
    API_KEY = '9ZRVL8BMOc8BxZVDMtOJm80LtqvGZWGCaQYmZr4X' # Or None to use default in semantic_scholar_api

    # Part 1: Retrieve papers from topics (like retrieve_from_topics.py)
    TOPICS_INPUT_JSON_FILE = 'results/query_logs/query_analysis_20250611_113308.json'
    YEAR_FILTER_TOPIC_SEARCH = "2020-"
    PAPERS_PER_TOPIC = 3
    API_REQUEST_DELAY_S_TOPIC_SEARCH = 1

    # --- Dynamic Filename/Directory Configuration based on TOPICS_INPUT_JSON_FILE ---
    base_input_filename = os.path.basename(TOPICS_INPUT_JSON_FILE)
    timestamp_match = re.search(r'(\d{8}_\d{6})', base_input_filename)
    
    if timestamp_match:
        timestamp_str = timestamp_match.group(1)
        print(f"Extracted timestamp: {timestamp_str}")
    else:
        # Fallback if timestamp pattern is not found
        from datetime import datetime
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Warning: Could not extract timestamp from '{base_input_filename}'. Using current timestamp: {timestamp_str}")

    # This output file from Part 1 will be the input for Part 2
    OUTPUT_TOPIC_SEARCH_RESULTS_FILE = f"results/paper_data/topic_search_{timestamp_str}.json"

    # Part 2: Get recommendations based on papers from Part 1 (like get_recommended_for_dblp.py)
    RECOMMENDATIONS_LIMIT = 15
    # Input for this part is OUTPUT_TOPIC_SEARCH_RESULTS_FILE
    OUTPUT_FILE_RECOMMENDATIONS = f"results/paper_data/recommended_papers_{timestamp_str}.json"
    DOWNLOAD_DIRECTORY = f"data/papers/{timestamp_str}" 
    API_REQUEST_DELAY_S_RECOMMENDATIONS = 2

    print("="*30 + " STARTING MAIN WORKFLOW " + "="*30)

    # --- PART 1: RETRIEVE PAPERS FROM TOPICS ---
    print("\n" + "="*20 + " PART 1: RETRIEVING PAPERS FROM TOPICS " + "="*20)
    
    # 1.1 Extract Topics
    print(f"\nExtracting topics from JSON file: {TOPICS_INPUT_JSON_FILE}...")
    topics = data_processing.extract_topics_from_json(TOPICS_INPUT_JSON_FILE)
    
    if not topics:
        print("No topics found. Please check your JSON file. Exiting workflow.")
        exit(1)
    print(f"Topics to search: {topics}")
    
    # 1.2 Perform Searches for topics
    print(f"\nSearching Semantic Scholar for {len(topics)} topics (up to {PAPERS_PER_TOPIC} papers per topic)...")
    topic_search_results = semantic_scholar_api.search_papers_for_topics_bulk(
        topics=topics,
        api_key=API_KEY,
        year_filter=YEAR_FILTER_TOPIC_SEARCH,
        delay_seconds=API_REQUEST_DELAY_S_TOPIC_SEARCH,
        papers_per_topic=PAPERS_PER_TOPIC
    )
    
    # 1.3 Display Summary of topic search
    if topic_search_results:
        reporting.print_topic_search_summary(topic_search_results)
    else:
        print("\nNo search results were generated from topics. Exiting workflow.")
        exit(1) # Exit if no papers found, as Part 2 depends on this

    # 1.4 Save Topic Search Results
    if topic_search_results:
        file_io.save_data_to_json(topic_search_results, OUTPUT_TOPIC_SEARCH_RESULTS_FILE)
        print(f"\nTopic search results saved to '{OUTPUT_TOPIC_SEARCH_RESULTS_FILE}'")
    else:
        # This case should ideally be caught by the exit above, but as a safeguard:
        print(f"\nNo topic search results to save. Exiting workflow.")
        exit(1)

    print("\n" + "="*20 + " PART 1 COMPLETED " + "="*20)

    # --- PART 2: GET RECOMMENDATIONS BASED ON RETRIEVED PAPERS ---
    print("\n" + "="*20 + " PART 2: GETTING RECOMMENDATIONS " + "="*20)

    # 2.1 Setup Download Directory for recommendations
    try:
        os.makedirs(DOWNLOAD_DIRECTORY, exist_ok=True)
        print(f"\nDownload directory '{DOWNLOAD_DIRECTORY}' for recommendations is ready.")
    except OSError as e:
        print(f"Error creating directory {DOWNLOAD_DIRECTORY}: {e}")
        exit(1)

    # 2.2 Load Seed Paper IDs from Part 1's output
    print(f"\nLoading and extracting paper IDs from '{OUTPUT_TOPIC_SEARCH_RESULTS_FILE}'...")
    # The search_data here is the output from the topic search
    search_data_for_recommendations = file_io.load_json_file(OUTPUT_TOPIC_SEARCH_RESULTS_FILE)
    
    if not search_data_for_recommendations:
        print(f"Could not load data from '{OUTPUT_TOPIC_SEARCH_RESULTS_FILE}'. Exiting workflow.")
        exit(1)
        
    topic_paper_ids_for_recs = data_processing.extract_paper_ids_from_search_data(search_data_for_recommendations)
    
    if not topic_paper_ids_for_recs or not any(topic_paper_ids_for_recs.values()):
        print("No paper IDs found in the topic search data. Cannot proceed with recommendations. Exiting.")
        exit(1)
        
    print("\nPooling seed papers from all topics for recommendations...")
    all_seed_ids_for_recs = data_processing.pool_paper_ids(topic_paper_ids_for_recs)
    
    if not all_seed_ids_for_recs:
        print("Could not find any seed paper IDs to use for recommendations. Exiting.")
        exit(1)
    print(f"Found {len(all_seed_ids_for_recs)} unique seed paper IDs for recommendations.")

    # 2.3 Get Recommendations
    print(f"\nGetting {RECOMMENDATIONS_LIMIT} recommendations based on pooled seeds...")
    recommended_papers = semantic_scholar_api.get_recommendations(
        all_seed_ids_for_recs, 
        api_key=API_KEY, 
        limit=RECOMMENDATIONS_LIMIT
    )
    
    # 2.4 Fetch Details & Download PDFs for recommendations
    if recommended_papers:
        semantic_scholar_api.fetch_batch_paper_details(
            recommended_papers, 
            api_key=API_KEY,
            delay_seconds=API_REQUEST_DELAY_S_RECOMMENDATIONS
        )

        print("\nAttempting to download PDFs for recommended papers...")
        for paper in recommended_papers:
            title = paper.get('title')
            paper_id = paper.get('paperId')
            
            pdf_url = data_processing.extract_pdf_url_from_paper_details(paper)
            paper['derivedPdfUrl'] = pdf_url # Store for record-keeping

            if pdf_url:
                print(f"  Processing recommended paper for download: '{title}' (ID: {paper_id})")
                file_io.download_pdf(
                    title=title,
                    paper_id=paper_id,
                    pdf_url=pdf_url,
                    download_dir=DOWNLOAD_DIRECTORY,
                    sanitize_func=data_processing.sanitize_filename
                )
            else:
                status = "Open Access" if paper.get('isOpenAccess') else "Not Open Access or Link not Found"
                print(f"  No downloadable PDF URL found for recommended paper: '{title}' (ID: {paper_id}). Status: {status}")
    else:
        print("\nNo recommendations were generated. Skipping detail fetching and PDF download for recommendations.")

    # 2.5 Process and Save Recommendation Results
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
        print("\nProcess finished, but no recommendations were generated or processed in Part 2.")

    print("\n" + "="*20 + " PART 2 COMPLETED " + "="*20)
    print("\n" + "="*30 + " MAIN WORKFLOW COMPLETED " + "="*30)