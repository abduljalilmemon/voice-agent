import json
import os

from dotenv import load_dotenv

load_dotenv()

# loaded once at import -- shared by the API and the agent graph
with open("data/business_config.json") as f:
    BIZ_CONFIG = json.load(f)

# who gets booking notifications -- kept in .env, not in the repo
OWNER_EMAIL = os.getenv("OWNER_EMAIL")

# where runtime state (conversation memory, bookings) lives -- a volume in Docker
STORAGE_DIR = os.getenv("STORAGE_DIR", ".")

def storage_path(filename: str) -> str:
    os.makedirs(STORAGE_DIR, exist_ok=True)
    return os.path.join(STORAGE_DIR, filename)
