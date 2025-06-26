import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from .agents.config import GEMINI_API_KEY, DEFAULT_MODEL

def gemini_llm_runner(prompt: str) -> str:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(DEFAULT_MODEL)
    generation_config = GenerationConfig(temperature=0)
    response = model.generate_content(prompt, generation_config=generation_config)
    return response.text

def get_llm_runner():
    """
    Central place to select the LLM runner for the whole backend.
    Returns the Gemini LLM runner by default.
    """
    return gemini_llm_runner 