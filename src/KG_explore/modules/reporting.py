import os

def print_recommendations_summary(papers, download_dir, sanitize_func):
    """
    Print a summary of a list of recommended papers.
    """
    print("\n" + "="*60)
    print("RECOMMENDATIONS SUMMARY")
    print("="*60)
    
    if not papers:
        print("No recommended papers to summarize.")
        return

    print(f"Total recommended papers processed: {len(papers)}")
    
    print("\nTop 5 recommended papers (by citation count):")
    for i, paper in enumerate(papers[:5], 1):
        title = paper.get('title', 'No title')
        citations_count = paper.get('citationCount', 0) # Actual citation count
        pub_date = paper.get('publicationDate', 'Unknown Date')
        
        references_list = paper.get('references') # This might be None
        num_refs = len(references_list) if references_list is not None else 0

        citations_list = paper.get('citations') # This might be None for 'cited by'
        num_cites_by = len(citations_list) if citations_list is not None else 0
        
        pdf_status = "Not Found / Not Open Access"
        if paper.get('derivedPdfUrl'):
            filename = sanitize_func(title) + ".pdf"
            file_path = os.path.join(download_dir, filename)
            if os.path.exists(file_path):
                pdf_status = "Downloaded"
            else:
                pdf_status = "Available (Download Attempted/Failed)"
        elif paper.get('isOpenAccess'):
             pdf_status = "Open Access (No Direct PDF Link Found for Download)"

        print(f"  {i}. {title}")
        print(f"     Citations: {citations_count} | Date: {pub_date} | Refs: {num_refs} | Cited By: {num_cites_by} | PDF: {pdf_status}")

def print_topic_search_summary(results):
    """
    Print a summary of topic search results.
    
    Args:
        results (dict): Search results dictionary where keys are topics
                        and values are API response dictionaries.
    """
    print("\n" + "="*50)
    print("TOPIC SEARCH RESULTS SUMMARY")
    print("="*50)
    
    if not results:
        print("No search results to summarize.")
        return

    for topic, result_data in results.items():
        print(f"\nTopic: {topic}")
        if result_data and 'data' in result_data and result_data['data']:
            papers = result_data['data']
            print(f"  Papers found: {len(papers)}")
            
            # Sort papers by citationCount if available, descending
            # S2 API search results are often pre-sorted by relevance, but we can re-sort if needed
            # papers.sort(key=lambda p: p.get('citationCount', 0), reverse=True)

            print("  Top papers (as returned by API):")
            for i, paper in enumerate(papers[:3], 1): # Show top 3 or fewer if less found
                title = paper.get('title', 'No title')
                pub_date = paper.get('publicationDate', 'Unknown date')
                citations = paper.get('citationCount', 0) # Ensure this field was requested
                print(f"    {i}. {title} (Date: {pub_date}, Citations: {citations})")
        elif result_data and 'data' in result_data and not result_data['data']:
            print("  No papers found for this topic.")
        else:
            print("  No results or API error for this topic.")