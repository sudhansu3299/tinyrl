# control_api.py
from fastapi import FastAPI, Request
import subprocess
import signal
import psutil
import logging

app = FastAPI()
logger = logging.getLogger(__name__)

@app.post("/control")
async def control_agent(request: Request):
    data = await request.json()
    action = data.get("action")
    target = data.get("target")

    logger.warning(f"⚠️ Received action: {action} -> {target}")

    if action == "kill_process":
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                if proc.info['name'] == target:
                    proc.send_signal(signal.SIGTERM)
                    logger.info(f"✅ Killed process {target} (PID: {proc.pid})")
                    return {"status": "killed", "pid": proc.pid}
            return {"status": "not_found", "reason": f"No process named {target}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    elif action == "restart_service":
        try:
            subprocess.run(["systemctl", "restart", target], check=True)
            return {"status": "restarted", "service": target}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    return {"status": "ignored", "reason": "Unsupported action"}
