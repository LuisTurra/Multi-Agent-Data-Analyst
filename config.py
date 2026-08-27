import os
from dotenv import load_dotenv

load_dotenv()

# Configurações
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEFAULT_MODEL =  "GROQ_MODEL","openai/gpt-oss-120b"  # Rápido e bom no Groq
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"