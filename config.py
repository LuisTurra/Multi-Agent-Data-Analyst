import os
from dotenv import load_dotenv

load_dotenv()

# Configurações
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEFAULT_MODEL = "llama-3.3-70b-versatile"  # Rápido e bom no Groq
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"