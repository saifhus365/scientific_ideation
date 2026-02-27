import json
from datetime import datetime
from pathlib import Path
from query_analyzer import QueryAnalyzer

def save_query_result(query: str, result: dict, results_dir: Path) -> None:
    """Save query and its analysis result to a JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"query_analysis_{timestamp}.json"
    
    # Create results directory if it doesn't exist
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data to save
    data = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "analysis": {
            "topics": result.topics,
            "timeline": {
                "start_date": result.timeline.start_date,
                "end_date": result.timeline.end_date,
                "specific_year": result.timeline.specific_year
            },
            "intention": result.intention
        }
    }
    
    # Save to file
    with open(results_dir / filename, 'w') as f:
        json.dump(data, f, indent=2)

def main():
    # Initialize analyzer
    analyzer = QueryAnalyzer()
    
    # Set up results directory
    results_dir = Path("/Users/husainsaif/Desktop/thesis-saif/results/query_logs")
    
    # Example query
    query = input('Query: ')
    
    try:
        # Analyze query
        result = analyzer.analyze_query(query)
        
        # Save results
        save_query_result(query, result, results_dir)
        
        # Print results
        print("\nAnalysis Results:")
        print(f"Topics: {result.topics}")
        print(f"Timeline: {result.timeline}")
        print(f"User Intention: {result.intention}")
        print(f"\nResults saved to: {results_dir}")
        
    except Exception as e:
        print(f"Error analyzing query: {str(e)}")

if __name__ == "__main__":
    main()