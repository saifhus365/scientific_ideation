import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import euclidean_distances
from literature_review.tools import search_papers_by_keyword, get_paper_details

# --- Configuration ---
BPAST_CUTOFF_YEAR = 2023
TOP_K_SIMILAR = 5
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_paper_embeddings(papers):
    """Generates embeddings for the abstracts of a list of papers."""
    abstracts = [p.get('abstract', '') for p in papers if p.get('abstract')]
    if not abstracts:
        return []
    return model.encode(abstracts, convert_to_tensor=True)

def calculate_dissimilarity(generated_embedding, historical_embeddings):
    """Calculates the average Euclidean distance to the top K most similar embeddings."""
    if len(historical_embeddings) == 0:
        return 0
    
    # Ensure generated_embedding is a 2D array for distance calculation
    if len(generated_embedding.shape) == 1:
        generated_embedding = generated_embedding.reshape(1, -1)

    distances = euclidean_distances(
        generated_embedding.cpu().numpy(), 
        historical_embeddings.cpu().numpy()
    )[0]
    
    # Get the distances of the top K most similar (smallest distance)
    top_k_distances = np.sort(distances)[:TOP_K_SIMILAR]
    
    return float(np.mean(top_k_distances))

def calculate_contemporary_impact(similar_papers):
    """Calculates the average citation count of a list of papers."""
    citation_counts = [p.get('citationCount', 0) for p in similar_papers]
    return np.mean(citation_counts) if citation_counts else 0

def get_similar_papers(query, year_start=None, year_end=None, limit=5):
    """Finds papers similar to a query within a given year range."""
    query_str = f"{query}"
    if year_start and year_end:
        query_str += f" year:{year_start}-{year_end}"
    elif year_start:
        query_str += f" year:{year_start}-"
    elif year_end:
        query_str += f" year:-{year_end}"
        
    return search_papers_by_keyword(query_str, limit=limit)

def calculate_novelty_metrics(generated_idea: dict, discovered_papers: list):
    """
    Calculates the full suite of novelty metrics for a single generated idea.
    """
    print(f"\n--- Calculating Novelty Metrics for Idea: '{generated_idea['title']}' ---")
    
    # 1. Generate embedding for the new idea
    idea_text = f"{generated_idea['title']} {generated_idea['description']}"
    idea_embedding = model.encode([idea_text], convert_to_tensor=True)
    
    # 2. Separate discovered papers into Bpast and Bcon
    bpast_papers = [p for p in discovered_papers if p.get('year') and p['year'] < BPAST_CUTOFF_YEAR]
    bcon_papers = [p for p in discovered_papers if p.get('year') and p['year'] >= BPAST_CUTOFF_YEAR]

    print(f"  - Bpast papers (before {BPAST_CUTOFF_YEAR}): {len(bpast_papers)}")
    print(f"  - Bcon papers ({BPAST_CUTOFF_YEAR} and after): {len(bcon_papers)}")

    # 3. Get embeddings for Bpast and Bcon papers
    bpast_embeddings = get_paper_embeddings(bpast_papers)
    bcon_embeddings = get_paper_embeddings(bcon_papers)

    # 4. Calculate Historical Dissimilarity (HD)
    hd = calculate_dissimilarity(idea_embedding, bpast_embeddings)
    print(f"  - Historical Dissimilarity (HD): {hd:.4f}")
    
    # 5. Calculate Contemporary Dissimilarity (CD)
    cd = calculate_dissimilarity(idea_embedding, bcon_embeddings)
    print(f"  - Contemporary Dissimilarity (CD): {cd:.4f}")

    # 6. Find top 5 similar contemporary papers for CI
    if bcon_papers:
        # Calculate cosine similarities to find the most similar papers
        cos_scores = util.pytorch_cos_sim(idea_embedding, bcon_embeddings)[0]
        top_k_indices = np.argsort(-cos_scores.cpu().numpy())[:TOP_K_SIMILAR]
        top_k_con_papers = [bcon_papers[i] for i in top_k_indices]
        
        # 7. Calculate Contemporary Impact (CI)
        ci = calculate_contemporary_impact(top_k_con_papers)
        print(f"  - Contemporary Impact (CI): {ci:.4f}")
    else:
        ci = 0
        print("  - No contemporary papers found to calculate CI.")

    # 8. Calculate Overall Novelty (ON)
    # Avoid division by zero
    if cd > 0:
        on = (hd * ci) / cd
    else:
        on = 0
    print(f"  - Overall Novelty (ON): {on:.4f}")
    
    return {
        "historical_dissimilarity": hd,
        "contemporary_dissimilarity": cd,
        "contemporary_impact": ci,
        "overall_novelty": on
    }

if __name__ == "__main__":
    # This is an example of how you would run the novelty calculation.
    # You would need to load your generated ideas and the literature review report.
    
    # Example Generated Idea
    sample_idea = {
        "title": "Using Multi-Agent Systems for Quantum Entanglement Simulation",
        "description": "A novel approach to simulate quantum entanglement by modeling particles as autonomous agents that communicate and update their states based on local rules, potentially offering a more scalable and computationally efficient method than traditional matrix-based simulations."
    }

    # You would load this from your `lit_review_report_*.json` files
    # For this example, we'll fetch some papers directly
    print("Fetching sample papers for demonstration...")
    discovered_papers = search_papers_by_keyword("Multi-Agent Systems AND Quantum Computing", limit=100)
    
    if discovered_papers:
        novelty_scores = calculate_novelty_metrics(sample_idea, discovered_papers)
        
        print("\n--- Final Novelty Scores ---")
        for metric, score in novelty_scores.items():
            print(f"{metric}: {score:.4f}")
    else:
        print("Could not fetch any papers to run the example.")