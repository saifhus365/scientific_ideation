import nltk
from nltk.corpus import stopwords
import string
import json 
from tqdm import tqdm 
from collections import Counter
import numpy as np
import pandas as pd
import argparse
import os
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt

def plot_string_occurrences(strings_list):
    # Count occurrences of each string
    occurrences = Counter(strings_list)
    
    # Count how many strings have each occurrence count
    count_of_occurrences = Counter(occurrences.values())
    
    # Extracting the data for plotting
    x = sorted(count_of_occurrences.keys())
    y = [count_of_occurrences[occ] for occ in x]
    
    # Plotting the data
    plt.figure(figsize=(10, 6))
    plt.bar(x, y, color='skyblue')
    plt.xlabel('Number of Occurrences')
    plt.ylabel('Number of Strings')
    plt.title('Frequency of String Occurrences')
    plt.xticks(x)
    plt.grid(axis='y')
    plt.show()

def process_text(input_text, tokenize=False):
    # Define the list of stopwords
    stop_words = set(stopwords.words('english'))
    
    # Lowercase the input text
    lowercased_text = input_text.lower()
    
    # Remove punctuation from the text
    no_punctuation_text = lowercased_text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize the text into words
    words = no_punctuation_text.split()
    
    # Remove stopwords
    filtered_words = [word for word in words if word not in stop_words]
    
    # Join the filtered words back into a single string
    processed_text = ' '.join(filtered_words)

    if tokenize:
        return set(filtered_words)
    else:
        return processed_text

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def find_representative_paper(cluster, similarity_matrix, labels):
    cluster_indices = [i for i, label in enumerate(labels) if label == cluster]
    cluster_sims = similarity_matrix[cluster_indices][:, cluster_indices]
    avg_sims = cluster_sims.mean(axis=1)
    representative_index = cluster_indices[avg_sims.argmax()]
    return representative_index

def find_top_n_papers(representative_index, similarity_matrix, n=5):
    sims = similarity_matrix[representative_index]
    closest_indices = np.argsort(-sims)[:n]  # Sort in descending order and get top-n
    return closest_indices

def concatenate_idea(idea):
    # Updated to match the NovelIdea data model
    title = idea.get('title', '')
    description = idea.get('description', '')
    reasoning = idea.get('reasoning', '')
    return f"{title}\n{description}\n{reasoning}"

def concatenate_idea_with_abstract(idea):
    """Concatenates the title and abstract of an idea for embedding."""
    title = idea.get('title', '')
    abstract = idea.get('abstract', '')
    return f"{title}\n{abstract}"

def run_deduplication(final_state: dict, project_root_path: str, run_timestamp: str, similarity_threshold=0.8):
    """
    Performs deduplication on the final list of generated ideas.
    """
    print("\n--- Stage 7: Deduplicating Final Ideas ---")
    
    final_ideas_obj = final_state.get("final_ideas")

    if not final_ideas_obj:
        print("Warning: No final_ideas object found in the state to deduplicate.")
        return None

    # Handle both Pydantic objects and dictionaries
    if hasattr(final_ideas_obj, 'final_ideas'):
        ideas = final_ideas_obj.final_ideas
    elif isinstance(final_ideas_obj, dict):
        ideas = final_ideas_obj.get("final_ideas", [])
    else:
        ideas = []

    if not ideas:
        print("No final ideas with abstracts to deduplicate.")
        return None

    # Convert Pydantic models to dictionaries for consistent processing
    ideas = [idea.dict() if hasattr(idea, 'dict') else idea for idea in ideas]


    print(f"Original number of ideas: {len(ideas)}")
    print(f"Using similarity threshold: {similarity_threshold}")

    # 1. Concatenate and encode all ideas
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", token=False)
    concatenated_ideas = [concatenate_idea_with_abstract(idea) for idea in ideas]
    embeddings = model.encode(concatenated_ideas, convert_to_tensor=True)


    # 2. Calculate the similarity matrix
    similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings).cpu().numpy()

    # 3. Filter out similar ideas
    final_indices = []
    filtered_out_indices = []
    for i in range(len(ideas)):
        if i not in filtered_out_indices:
            final_indices.append(i)
            # Find and filter out subsequent similar ideas
            for j in range(i + 1, len(ideas)):
                if j not in filtered_out_indices and similarity_matrix[i][j] > similarity_threshold:
                    filtered_out_indices.append(j)
    
    deduplicated_ideas = [ideas[i] for i in final_indices]
    print(f"Number of ideas after deduplication: {len(deduplicated_ideas)}")
    
    # 4. Save the deduplicated ideas to a new file
    output_dir = project_root_path / "results" / "final_reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"deduplicated_ideas_{run_timestamp}.json"
    
    report = {
        "original_query": final_state.get("initial_query"),
        "intention": final_state.get("intention"),
        "topics": final_state.get("topics"),
        "similarity_threshold": similarity_threshold,
        "original_idea_count": len(ideas),
        "deduplicated_idea_count": len(deduplicated_ideas),
        "final_ideas": deduplicated_ideas
    }
    
    return deduplicated_ideas