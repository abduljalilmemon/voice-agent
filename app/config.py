import json
import os

# loaded once at import -- shared by the API and the agent graph
with open("data/business_config.json") as f:
    BIZ_CONFIG = json.load(f)

# where runtime state (conversation memory, bookings) lives -- a volume in Docker
STORAGE_DIR = os.getenv("STORAGE_DIR", ".")

def storage_path(filename: str) -> str:
    os.makedirs(STORAGE_DIR, exist_ok=True)
    return os.path.join(STORAGE_DIR, filename)
