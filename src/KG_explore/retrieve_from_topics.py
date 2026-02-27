
import os # For path joining if needed for output file
from .modules import data_processing
from .modules import semantic_scholar_api
from .modules import reporting
from .modules import file_io # For saving results



# Main execution
if __name__ == "__main__":
    # Configuration
    TOPICS_INPUT_JSON_FILE = 'results/query_logs/query_analysis_20250611_113308.json'
    # API_KEY can be set to None to use the default in semantic_scholar_api.py,
    # or provide a specific key here.
    API_KEY = None # Uses DEFAULT_API_KEY from semantic_scholar_api.py
    # API_KEY = 'YOUR_SEMANTIC_SCHOLAR_API_KEY' # Or set one explicitly
    
    YEAR_FILTER = "2020-"  # Search from 2020 onwards
    PAPERS_PER_TOPIC = 3   # Max papers to retrieve per topic
    API_REQUEST_DELAY_S = 1 # Delay between API calls for different topics
    
    # Construct output file path dynamically or keep it fixed
    # Example: base_filename = os.path.basename(TOPICS_INPUT_JSON_FILE).replace('.json', '')
    # OUTPUT_SEARCH_RESULTS_FILE = f"results/paper_data/semantic_scholar_topic_results_{base_filename}.json"
    OUTPUT_SEARCH_RESULTS_FILE = "results/paper_data/semantic_scholar_results_from_topics.json"


    # --- 1. Extract Topics ---
    print(f"Extracting topics from JSON file: {TOPICS_INPUT_JSON_FILE}...")
    topics = data_processing.extract_topics_from_json(TOPICS_INPUT_JSON_FILE)
    
    if not topics:
        print("No topics found. Please check your JSON file or define topics manually.")
        exit(1)
    
    print(f"Topics to search: {topics}")
    
    # --- 2. Perform Searches ---
    print(f"\nSearching Semantic Scholar for {len(topics)} topics (up to {PAPERS_PER_TOPIC} papers per topic)...")
    search_results = semantic_scholar_api.search_papers_for_topics_bulk(
        topics=topics,
        api_key=API_KEY,
        year_filter=YEAR_FILTER,
        delay_seconds=API_REQUEST_DELAY_S,
        papers_per_topic=PAPERS_PER_TOPIC
    )
    
    # --- 3. Display Summary ---
    if search_results:
        reporting.print_topic_search_summary(search_results)
    else:
        print("\nNo search results were generated.")

    # --- 4. Save Results ---
    if search_results:
        file_io.save_data_to_json(search_results, OUTPUT_SEARCH_RESULTS_FILE)
    else:
        print(f"\nNo results to save.")
    
    print(f"\nTopic search completed! Check '{OUTPUT_SEARCH_RESULTS_FILE}' for detailed results if any were found.")