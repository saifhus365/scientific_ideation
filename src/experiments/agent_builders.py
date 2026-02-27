import os
import json
from pathlib import Path
import chromadb

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv


load_dotenv()
PROMPTS_PATH = Path(__file__).parent / "prompts.json"
PROMPTS_GEN_ONLY_PATH = Path(__file__).parent / "prompts_generate_only.json"

# Load all prompts from both files
with open(PROMPTS_PATH, 'r') as f:
    prompts = json.load(f)
with open(PROMPTS_GEN_ONLY_PATH, 'r') as f:
    prompts.update(json.load(f)) # Merge the new prompts into the main dict


def build_agent(llm: ChatMistralAI, prompt_key: str, output_model: BaseModel):
    """Generic agent builder."""
    prompt_template = prompts[prompt_key]
    prompt = ChatPromptTemplate.from_template(prompt_template)
    return prompt | llm.with_structured_output(output_model)