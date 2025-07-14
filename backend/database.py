# database.py
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from typing import Generator


DATABASE_URL = "sqlite:///./metrics.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False)
Base = declarative_base()

class Metric(Base):
    __tablename__ = "metrics"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime)
    host_id = Column(String)
    cpu_percent = Column(Float)
    memory_percent = Column(Float)
    disk_percent = Column(Float)
    top_processes = Column(JSON)

class Decision(Base):
    __tablename__ = "decisions"

    id = Column(Integer, primary_key=True, index=True)
    action = Column(String, nullable=False)
    target = Column(String, nullable=False)
    reason = Column(String)
    host_id = Column(String)
    status = Column(String, default="pending")
    timestamp = Column(DateTime)


def init_db():
    Base.metadata.create_all(bind=engine)

def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

