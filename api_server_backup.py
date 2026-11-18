
from fastapi import FastAPI
from datetime import datetime

app = FastAPI(
    title="AUTONOMOUS Driver Monitoring API",
    version="1.0.0"
)

@app.get("/")
def home():
    return {
        "status": "connected",
        "device": "AUTONOMOUS",
        "message": "Python server is running"
    }

@app.get("/health")
def health():
    return {
        "connection": "active",
        "camera": "waiting",
        "ai_engine": "ready",
        "timestamp": datetime.now().isoformat()
    }
