import os
import json
from pathlib import Path
import chromadb

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain.output_parsers.pydantic import PydanticOutputParser
# --- Configuration ---
load_dotenv()
CHROMA_PERSIST_DIR = Path(__file__).resolve().parent.parent.parent / "chroma_db"
CHROMA_COLLECTION_BASE_NAME = "lit_review_papers"
PROMPTS_PATH = Path(__file__).parent / "prompts.json"
NUM_DOCS_TO_RETRIEVE = 3 # Top N documents to retrieve for each debater

# Load all prompts
with open(PROMPTS_PATH, 'r') as f:
    prompts = json.load(f)

# --- Tool for Data Retrieval ---

def get_retriever_tool(run_timestamp: str):
    """
    Creates a retriever function for a specific ChromaDB collection.
    """
    def retrieve_from_chroma(query: str) -> str:
        """
        Retrieves relevant documents from the ChromaDB collection for the given run.
        """
        print(f"Retrieving documents for query: '{query}'")
        try:
            client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
            collection_name = f"{CHROMA_COLLECTION_BASE_NAME}_{run_timestamp}"

            # Check if the collection exists before trying to access it
            existing_collections = [c.name for c in client.list_collections()]
            if collection_name not in existing_collections:
                error_msg = (
                    f"ChromaDB collection '{collection_name}' not found. "
                    f"Please ensure you have run the data indexing workflow ('src/data_indexing/indexer.py') "
                    f"for the timestamp '{run_timestamp}' before running the debate."
                )
                print(f"Error: {error_msg}")
                return error_msg
            
            collection = client.get_collection(name=collection_name)
            
            results = collection.query(
                query_texts=[query],
                n_results=NUM_DOCS_TO_RETRIEVE
            )
            
            documents = results.get("documents", [[]])[0]
            if not documents:
                return "No relevant documents found in the database."

            # Format the documents for context
            formatted_docs = "\n\n".join(
                [f"--- Document {i+1} ---\n{doc}" for i, doc in enumerate(documents)]
            )
            return formatted_docs
        except Exception as e:
            print(f"Error retrieving from ChromaDB: {e}")
            return f"Error accessing the knowledge base: {e}"

    return retrieve_from_chroma

# --- Agent Creation Functions ---

def build_structured_agent(llm: ChatMistralAI,  prompt_key: str,  output_model: BaseModel,):
    """Generic agent builder."""
    prompt_template = prompts[prompt_key]
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    return prompt | llm.with_structured_output(output_model)



def build_raw_agent(llm: ChatMistralAI, prompt_key: str):
    """
    Builds a simple agent that returns a raw string response.
    """
    prompt_template = prompts.get(prompt_key)
    if not prompt_template:
        raise ValueError(f"Prompt key '{prompt_key}' not found in prompts.json")
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    return prompt | llm | (lambda msg: msg.content)


def build_debater_agent(llm: ChatMistralAI, output_model: BaseModel):
    """Builds the debating agent (without retriever attached yet)."""
    prompt_template = prompts["debating_agent_prompt"]
    prompt = ChatPromptTemplate.from_template(prompt_template)
    return prompt | llm.with_structured_output(output_model)
