FROM python:3.10-slim

WORKDIR /app

COPY . .
RUN pip install --no-cache-dir -r requirements.txt

ENV DB_PATH=/app/backend/metrics.db

CMD ["python", "decision_engine.py"]
