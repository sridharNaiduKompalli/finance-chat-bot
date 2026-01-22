from huggingface_hub import InferenceClient
import google.generativeai as genai
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Get API keys
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Import model configs
try:
    from config import GRANITE_MODELS, OTHER_MODELS
except ImportError:
    # Fallback if config import fails
    GRANITE_MODELS = {
        "granite_code": "ibm-granite/granite-3b-code-base",
        "granite_instruct": "ibm-granite/granite-7b-instruct",
    }
    OTHER_MODELS = {
        "llama2": "meta-llama/Llama-2-7b-chat-hf",
        "falcon": "tiiuae/falcon-7b-instruct",
    }

class MultiLLMManager:
    def __init__(self):
        # Initialize Hugging Face client
        if HF_API_KEY:
            self.hf_client = InferenceClient(token=HF_API_KEY)
        else:
            self.hf_client = None
            print("Warning: Hugging Face API key not found")
        
        self.models = {**GRANITE_MODELS, **OTHER_MODELS}
        
        # Initialize Gemini
        if GOOGLE_API_KEY:
            try:
                genai.configure(api_key=GOOGLE_API_KEY)
                # Try different model names
                model_names = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
                self.gemini_model = None
                
                for model_name in model_names:
                    try:
                        self.gemini_model = genai.GenerativeModel(model_name)
                        print(f"✅ Successfully loaded Gemini: {model_name}")
                        break
                    except Exception as e:
                        print(f"❌ Failed to load {model_name}: {e}")
                        continue
                        
                if not self.gemini_model:
                    print("⚠️ Warning: Could not initialize any Gemini model")
            except Exception as e:
                print(f"Error configuring Gemini: {e}")
                self.gemini_model = None
        else:
            self.gemini_model = None
            print("Warning: Google API key not found")

    @st.cache_data
    def query_huggingface_model(_self, model_name, prompt, max_tokens=200):
        """Query Hugging Face models"""
        try:
            if not _self.hf_client:
                return "Hugging Face API key not configured"
                
            if model_name not in _self.models:
                return "Model not found"
            
            response = _self.hf_client.text_generation(
                prompt=prompt,
                model=_self.models[model_name],
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True
            )
            return response
        except Exception as e:
            return f"Error with {model_name}: {str(e)}"

    def query_gemini(self, prompt):
        """Query Gemini model"""
        try:
            if not self.gemini_model:
                return "Gemini API key not configured or model not available"
            
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Gemini Error: {str(e)}"

    def get_response(self, model_name, prompt):
        """Main method to get response from any model"""
        if model_name == "gemini":
            return self.query_gemini(prompt)
        else:
            return self.query_huggingface_model(model_name, prompt)