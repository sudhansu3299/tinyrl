from dotenv import load_dotenv
import os

load_dotenv()  # load from .env

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_PATH = os.getenv("DB_PATH", "/app/backend/metrics.db")
AGENT_CONTROL_URL = os.getenv("AGENT_CONTROL_URL", "http://cloud-agent:5000/control")
USE_LOCAL_AI = os.getenv("USE_LOCAL_AI")