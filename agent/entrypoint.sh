#!/bin/bash
set -e

echo "🚀 Starting Cloud Agent for host: $HOST_ID"

#start control api to get actions!
uvicorn control_api:app --host 0.0.0.0 --port 9000 &

# Use -u for unbuffered + rely on PYTHONUNBUFFERED
exec python -u agent.py
