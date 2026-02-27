import pdfplumber
import json
import re
from pathlib import Path

def clean_and_split_into_paragraphs(text_block: str) -> list[str]:
    """
    Cleans a block of extracted text and splits it into a list of paragraphs.
    """
    # 1. Remove common citation and footnote artifacts.
    # Removes [1], [2,3], [ABC+12], etc.
    text_block = re.sub(r'\[[^\]]+\]', '', text_block)
    # Removes unicode symbols often used for footnotes or in math.
    text_block = re.sub(r'[\u2217\u2020\u2021\u00b9\u00b2\u00b3\uf0b7\u25b6]', '', text_block)

    # 2. Handle hyphenation across line breaks.
    text_block = re.sub(r'-\n', '', text_block)

    # 3. Split the text block into paragraphs.
    # A paragraph is assumed to be a block of text separated by two or more newlines.
    paragraphs = re.split(r'\n{2,}', text_block)
    
    cleaned_paragraphs = []
    for para in paragraphs:
        # Within each paragraph, replace single newlines with spaces to join wrapped lines.
        cleaned_para = para.replace('\n', ' ').strip()
        # Consolidate multiple spaces into one.
        cleaned_para = re.sub(r'\s+', ' ', cleaned_para)
        
        # Add the paragraph if it's not just whitespace.
        if cleaned_para:
            cleaned_paragraphs.append(cleaned_para)
            
    return cleaned_paragraphs

def extract_structured_content(pdf_path: Path) -> dict:
    """
    Extracts text from a PDF, structuring it by sections and paragraphs.
    """
    print(f"-> Processing: {pdf_path.name}")
    content = {}
    current_section_key = "header"
    text_buffer = []

    # Get clean paper title from filename to help filter page headers
    clean_paper_title = re.sub(r'[-_]', ' ', pdf_path.stem).lower()

    # Patterns for canonical sections
    section_patterns = {
        "abstract": re.compile(r"^\s*abstract\s*$", re.IGNORECASE),
        "introduction": re.compile(r"^\s*(1\s*\.?\s*)?introduction\s*$", re.IGNORECASE),
        "related_work": re.compile(r"^\s*(\d+\s*\.?\s*)?(related\s+work|background|preliminaries)\s*$", re.IGNORECASE),
        "methodology": re.compile(r"^\s*(\d+\s*\.?\s*)?(methodology|methods|approach|materials\s+and\s+methods)\s*$", re.IGNORECASE),
        "experiments": re.compile(r"^\s*(\d+\s*\.?\s*)?(experiments|results|evaluation)\s*$", re.IGNORECASE),
        "discussion": re.compile(r"^\s*(\d+\s*\.?\s*)?discussion\s*$", re.IGNORECASE),
        "conclusion": re.compile(r"^\s*(\d+\s*\.?\s*)?(conclusion|conclusions)\s*$", re.IGNORECASE),
        "acknowledgements": re.compile(r"^\s*acknowledgements\s*$", re.IGNORECASE),
        "references": re.compile(r"^\s*references\s*$", re.IGNORECASE),
    }
    # A more constrained pattern for numbered sections/subsections.
    generic_section_pattern = re.compile(r"^\s*(\d{1,2}(\.\d+)*)\s+(.*)")

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text(x_tolerance=2, y_tolerance=2)
                if not page_text:
                    continue

                lines = page_text.split('\n')
                for line in lines:
                    stripped_line = line.strip()
                    
                    if not stripped_line or stripped_line.isdigit():
                        continue

                    new_section_title = None
                    # First, check for specific, canonical sections
                    for section_name, pattern in section_patterns.items():
                        if pattern.match(stripped_line):
                            new_section_title = stripped_line.strip()
                            break
                    
                    # If not a canonical one, check for a generic numbered section
                    if not new_section_title:
                        match = generic_section_pattern.match(stripped_line)
                        if match:
                            title_part = match.group(3)
                            
                            # --- Heuristic Checks to Validate Generic Sections ---
                            # 1. Reject if it looks like code or has odd characters.
                            if any(c in title_part for c in ['=', ';', '_', '(', ')', '{', '}']):
                                continue
                            
                            # 2. Reject if the line is excessively long.
                            if len(stripped_line) > 150:
                                continue

                            # 3. Reject if it looks like a page header (is long and very similar to paper title).
                            title_words = set(title_part.lower().split())
                            if len(title_words) > 4: # Only check longer titles
                                paper_title_words = set(clean_paper_title.split())
                                common_words = title_words.intersection(paper_title_words)
                                # If more than 4 words in common, likely a page header
                                if len(common_words) > 4:
                                    continue
                            
                            new_section_title = stripped_line.strip()

                    if new_section_title:
                        if text_buffer:
                            raw_text = "\n".join(text_buffer)
                            content[current_section_key] = clean_and_split_into_paragraphs(raw_text)
                        
                        text_buffer = []
                        current_section_key = new_section_title
                    else:
                        text_buffer.append(stripped_line)

            if text_buffer:
                raw_text = "\n".join(text_buffer)
                content[current_section_key] = clean_and_split_into_paragraphs(raw_text)

        content = {k: v for k, v in content.items() if v}

        print(f"   ... Found sections: {list(content.keys())}")
        return content

    except Exception as e:
        print(f"   !!! Error processing {pdf_path.name}: {e}")
        return {"error": str(e), "filename": pdf_path.name}


def process_directory(input_pdf_dir: Path, output_json_dir: Path):
    """
    Processes all PDFs in an input directory and saves structured JSONs to an output directory.
    """
    print("--- Starting PDF Processing Workflow ---")

    output_json_dir.mkdir(parents=True, exist_ok=True)
    print(f"Input PDF directory: {input_pdf_dir}")
    print(f"Output JSON directory is: {output_json_dir}")

    pdf_files = list(input_pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"Warning: No PDF files found in '{input_pdf_dir}'")
        return
        
    print(f"Found {len(pdf_files)} PDF(s) to process.\n")

    for pdf_path in pdf_files:
        structured_data = extract_structured_content(pdf_path)
        
        json_filename = pdf_path.with_suffix('.json').name
        output_path = output_json_dir / json_filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(structured_data, f, indent=4)
        
        print(f"   -> Saved cleaned data to {output_path}\n")

    print("--- PDF Processing Workflow Finished ---")


def main():
    """Main function for standalone execution."""
    project_root = Path(__file__).resolve().parent.parent.parent
    # NOTE: You may need to update this timestamp for standalone testing.
    timestamped_folder = "20250805_101008" 
    input_pdf_dir = project_root / "data" / "papers" / timestamped_folder
    output_json_dir = project_root / "data" / "processed_papers"
    process_directory(input_pdf_dir, output_json_dir)


if __name__ == "__main__":
    main() 