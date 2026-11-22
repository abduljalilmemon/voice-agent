import json

# loaded once at import -- shared by the API and the agent graph
with open("data/business_config.json") as f:
    BIZ_CONFIG = json.load(f)
