import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Model configurations
GRANITE_MODELS = {
    "granite_code": "ibm-granite/granite-3b-code-base",
    "granite_instruct": "ibm-granite/granite-7b-instruct",
}

OTHER_MODELS = {
    "llama2": "meta-llama/Llama-2-7b-chat-hf",
    "falcon": "tiiuae/falcon-7b-instruct",
}

# Streamlit configuration
STREAMLIT_CONFIG = {
    "page_title": "Personal Finance AI",
    "page_icon": "🤖",
    "layout": "wide"
}