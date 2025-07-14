# models.py
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class ProcessInfo(BaseModel):
    pid: int
    name: str
    cpu: float

class MetricIn(BaseModel):
    timestamp: datetime
    host_id: str
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    top_processes: List[ProcessInfo]

class DecisionOut(BaseModel):
    id: int
    action: str
    target: str
    reason: Optional[str]
    host_id: Optional[str]
    status: str
    timestamp: datetime

    class Config:
        orm_mode = True
