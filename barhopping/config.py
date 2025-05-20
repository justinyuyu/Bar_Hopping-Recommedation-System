import os
import yaml

# Get project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load YAML config
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "default.yml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Core settings
CITY = config["city"]
MAX_BARS = config["max_bars"]
MAX_PHOTOS = config["max_photos"]
MAX_REVS = config["max_reviews"]
TOP_K = config["top_k"]

# Database paths
BARS_DB = config["bars_db"]
QUERIES_DB = config["queries_db"]

# Model settings
GEMMA_MODEL = config["gemma_model"]
GRANITE_MODEL = config["granite_model"]

# API keys
HF_TOKEN = os.getenv("HF_TOKEN", config["hf_token"])
OPENAI_KEY = os.getenv("OPENAI_KEY", config["openai_key"])
