import os
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBI0dl_WlMuPsIWDc5YWI9u_yxquKD5Ytk")
DEFAULT_MODEL = "gemini-2.0-flash"
MAX_TOKENS = 2048 