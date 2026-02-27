import json
import os
import requests

def load_json_file(file_path):
    """Loads data from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{file_path}'.")
        return None
    except Exception as e:
        print(f"Error loading JSON file '{file_path}': {e}")
        return None

def save_data_to_json(data, output_file):
    """Saves data to a JSON file with indentation."""
    try:
        # Ensure the directory exists before trying to save the file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as file:
            json.dump(data, file, indent=2)
        print(f"Data saved to {output_file}")
    except Exception as e:
        print(f"Error saving data to JSON file '{output_file}': {e}")

def download_pdf(title, paper_id, pdf_url, download_dir, sanitize_func):
    """
    Downloads a paper from a URL and saves it to a directory using a sanitized title.

    Args:
        title (str): The title of the paper.
        paper_id (str): The unique ID of the paper, for logging.
        pdf_url (str): The direct URL to the PDF file.
        download_dir (str): The directory to save the PDF in.
        sanitize_func (function): Function to sanitize the filename.
    """
    if not pdf_url:
        print(f"    -> No PDF URL provided for paper {paper_id} ('{title}'). Skipping download.")
        return

    filename = sanitize_func(paper_id) + ".pdf"
    file_path = os.path.join(download_dir, filename)

    if os.path.exists(file_path):
        print(f"    -> Paper '{filename}' already downloaded. Skipping.")
        return

    try:
        print(f"    -> Downloading '{filename}' from {pdf_url}")
        response = requests.get(pdf_url, stream=True, timeout=30)
        response.raise_for_status()

        os.makedirs(download_dir, exist_ok=True) # Ensure directory exists
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"    -> Successfully saved to {file_path}")

    except requests.exceptions.RequestException as e:
        print(f"    -> FAILED to download paper {paper_id} ('{title}'): {e}")
    except Exception as e:
        print(f"    -> An unexpected error occurred during download for {paper_id} ('{title}'): {e}")