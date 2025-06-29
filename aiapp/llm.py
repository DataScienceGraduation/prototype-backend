import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

LLM_API_KEY = os.getenv("LLM_API_KEY")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")


def gemini_llm_runner(prompt: str) -> str:
    genai.configure(api_key=LLM_API_KEY)
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