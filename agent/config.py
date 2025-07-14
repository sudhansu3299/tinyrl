# config.py
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
HOST_ID = os.getenv("HOST_ID", "vm-1")
INTERVAL = int(os.getenv("INTERVAL", 10))  # seconds
