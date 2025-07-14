# Dockerfile
FROM python:3.11-slim

# 1) Ensure unbuffered output so print() / logging goes straight to stdout
ENV PYTHONUNBUFFERED=1

# 2) Install system deps and psutil
RUN apt-get update && \
    apt-get install -y gcc python3-dev && \
    pip install --no-cache-dir psutil requests

# 3) Copy your code
WORKDIR /app
COPY . .

# 4) Make entrypoint executable
RUN chmod +x entrypoint.sh

# 5) Run it
ENTRYPOINT ["./entrypoint.sh"]
