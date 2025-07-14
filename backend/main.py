# main.py
from fastapi import FastAPI, Request
from models import MetricIn
from database import init_db, SessionLocal, Metric
import logging
from fastapi import Depends
from sqlalchemy.orm import Session
from fastapi.responses import JSONResponse
import requests
from fastapi.middleware.cors import CORSMiddleware
from database import get_db, Decision
from datetime import datetime
from models import DecisionOut
from typing import List


logging.basicConfig(
    level=logging.INFO,  # 👈 Enable INFO level logs
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI()

# Init DB
init_db()

HOST_TO_AGENT = {
    "vm-1": "cloud-agent-vm1",
    "vm-2": "cloud-agent-vm2",
}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:8080"],  # Your frontend URLs
    allow_credentials=True,
    allow_methods=["*"],  # Or specify: ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    allow_headers=["*"],  # Or specify the headers you need
)


@app.get("/metrics/recent")
def get_recent_metrics(limit: int = 5, db: Session = Depends(get_db)):
    metrics = db.query(Metric).order_by(Metric.timestamp.desc()).limit(limit).all()
    result = []
    for m in metrics:
        result.append({
            "timestamp": m.timestamp.isoformat(),
            "host_id": m.host_id,
            "cpu": m.cpu_percent,
            "mem": m.memory_percent,
            "disk": m.disk_percent,
            "top_processes": m.top_processes
        })
    return JSONResponse(content=result)

@app.post("/metrics")
async def receive_metrics(metric: MetricIn):
    logging.info(f"📥 [{metric.timestamp}] Host: {metric.host_id} | CPU: {metric.cpu_percent}% | Top: {metric.top_processes[0].name if metric.top_processes else 'N/A'}")

    # Save to DB
    db = SessionLocal()
    db_metric = Metric(
        timestamp=metric.timestamp,
        host_id=metric.host_id,
        cpu_percent=metric.cpu_percent,
        memory_percent=metric.memory_percent,
        disk_percent=metric.disk_percent,
        top_processes=[p.dict() for p in metric.top_processes]
    )
    db.add(db_metric)
    db.commit()
    db.close()

    return {"status": "received"}

@app.post("/take-action")
async def take_action(request: Request, db: Session = Depends(get_db)):
    action_data = await request.json()
    host_id = action_data.get("host_id")

    # Save the action to the database
    decision = Decision(
        action=action_data.get("action"),
        target=action_data.get("target"),
        reason=action_data.get("reason"),
        host_id=host_id,
        status="pending",
        timestamp=datetime.utcnow()
    )
    db.add(decision)
    db.commit()
    db.refresh(decision)

    # Forward to agent
    agent_url = f"http://{HOST_TO_AGENT.get(host_id)}:9000/control"
    print("Sending action to the host:", agent_url)

    try:
        res = requests.post(agent_url, json=action_data, timeout=3)
        decision.status = "success"
        db.commit()
        return {
            "forwarded": True,
            "response": res.json(),
            "decision_id": decision.id
        }
    except Exception as e:
        decision.status = "failed"
        db.commit()
        return {
            "forwarded": False,
            "error": str(e),
            "decision_id": decision.id
        }

@app.get("/ai/decisions", response_model=List[DecisionOut])
def get_decisions(limit: int = 20, db: Session = Depends(get_db)):
    decisions = db.query(Decision).order_by(Decision.timestamp.desc()).limit(limit).all()
    logger.info("Sending Decisions to frontend: " + str(decisions))
    return  decisions