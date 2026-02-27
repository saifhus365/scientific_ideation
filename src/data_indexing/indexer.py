import chromadb
import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Configuration ---
# Where ChromaDB will store its persistent data
CHROMA_PERSIST_DIR = Path(__file__).resolve().parent.parent.parent / "chroma_db"
# The base name for collections. The final name will be suffixed with the run timestamp.
CHROMA_COLLECTION_BASE_NAME = "lit_review_papers"

# Configuration for splitting text into chunks
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=10000,
    chunk_overlap=2000,
    length_function=len,
    is_separator_regex=False,
)

def get_chroma_client() -> chromadb.PersistentClient:
    """Initializes and returns a persistent ChromaDB client."""
    return chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))

def index_run(run_timestamp: str):
    """
    Main function to process and index all papers from a specific workflow run.
    It reads the final report, processes each paper, chunks the content,
    and indexes it into ChromaDB.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    report_path = project_root / "results" / "final_reports" / f"lit_review_report_{run_timestamp}.json"
    processed_papers_dir = project_root / "data" / "processed_papers" / f"lit_review_papers_{run_timestamp}"

    print("--- Starting Data Indexing Workflow ---")
    print(f"ChromaDB persistence directory: {CHROMA_PERSIST_DIR}")
    
    if not report_path.exists():
        print(f"Error: Final report not found at {report_path}. Aborting.")
        return

    # 1. Load the main report JSON
    with open(report_path, 'r', encoding='utf-8') as f:
        report_data = json.load(f)
    
    all_papers = report_data.get("discovered_papers", [])
    if not all_papers:
        print("No papers discovered in the report. Nothing to index.")
        return

    print(f"Found {len(all_papers)} papers in report to process for indexing.")

    # 2. Prepare documents for ChromaDB
    all_chunks = []
    all_metadatas = []
    all_ids = []

    for i, paper in enumerate(all_papers):
        paper_id = paper.get("paperId", f"unknown_id_{i}")
        paper_title = paper.get("title", "Unknown Title")
        
        # Construct the path to the processed JSON file for the paper
        processed_file_path = processed_papers_dir / f"{paper_id}.json"

        content_source = ""
        chunks = []

        if processed_file_path.exists():
            # --- Primary Strategy: Use fully processed paper ---
            content_source = f"Processed file: {processed_file_path.name}"
            with open(processed_file_path, 'r', encoding='utf-8') as f:
                processed_data = json.load(f)
            
            full_text = ""
            # Combine all paragraphs from all sections into one text block
            for section_title, paragraphs in processed_data.items():
                if isinstance(paragraphs, list):
                    full_text += f"\n\n--- {section_title.upper()} ---\n\n" + " ".join(paragraphs)
            
            if full_text:
                chunks = TEXT_SPLITTER.split_text(full_text.strip())

        else:
            # --- Fallback Strategy: Use abstract and TLDR ---
            content_source = "Fallback (Abstract + TLDR)"
            abstract = paper.get("abstract", "")
            tldr = paper.get("tldr", {}).get("text", "") if paper.get("tldr") else ""
            
            fallback_text = f"Title: {paper_title}\n\nAbstract: {abstract}\n\nTLDR: {tldr}"
            
            if fallback_text.strip():
                chunks = TEXT_SPLITTER.split_text(fallback_text)

        if chunks:
            print(f"  -> Paper '{paper_id}' ({content_source}): Chunked into {len(chunks)} documents.")
            for chunk_idx, chunk_text in enumerate(chunks):
                all_chunks.append(chunk_text)
                
                # Create metadata for this chunk
                all_metadatas.append({
                    "paper_id": paper_id,
                    "title": paper_title,
                    "year": paper.get("year"),
                    "run_timestamp": run_timestamp,
                    "source": content_source,
                    "chunk_index": chunk_idx,
                })
                
                # Create a unique ID for this chunk
                all_ids.append(f"{paper_id}_chunk_{chunk_idx}")
        else:
            print(f"  -> Paper '{paper_id}': No content found to chunk.")

    # 3. Index the documents in ChromaDB
    if not all_chunks:
        print("\nNo text chunks were generated. Skipping database indexing.")
        print("--- Data Indexing Workflow Finished ---")
        return
        
    client = get_chroma_client()
    # Create a unique collection name for this specific run
    collection_name = f"{CHROMA_COLLECTION_BASE_NAME}_{run_timestamp}"
    collection = client.get_or_create_collection(name=collection_name)

    print(f"\nIndexing {len(all_chunks)} chunks into Chroma collection '{collection_name}'...")
    
    # ChromaDB's `add` is idempotent. If IDs already exist, they are updated.
    collection.add(
        documents=all_chunks,
        metadatas=all_metadatas,
        ids=all_ids
    )

    print(f"Successfully indexed {collection.count()} total documents in the collection.")
    print("--- Data Indexing Workflow Finished ---")


if __name__ == "__main__":
    # This allows you to run the indexing for a specific session directly.
    # You will need to replace this with a valid timestamp from your previous runs.
    # Look for a folder in 'results/final_reports' or 'data/processed_papers'.
    # For example: '20250809_094825'
    
    test_timestamp = "20250809_094825" # <--- IMPORTANT: CHANGE THIS TO YOUR RUN'S TIMESTAMP
    
    if test_timestamp == "CHANGE_ME":
        print("Error: Please open 'src/data_indexing/indexer.py' and set the 'test_timestamp' variable.")
    else:
        index_run(test_timestamp)
