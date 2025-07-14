import requests
import json
import time
import logging
import re
from config import OPENAI_API_KEY, AGENT_CONTROL_URL, USE_LOCAL_AI

from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam

client = OpenAI(api_key=OPENAI_API_KEY)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

AGENT_CONTROL_URL = "http://backend:8000/take-action"

def send_to_agent(decision, host_id):
    if decision["action"] == "none":
        return
    payload = {**decision, "host_id": host_id}
    try:
        res = requests.post(AGENT_CONTROL_URL, json=payload)
        logger.info(f"✅ Sent action to backend → agent: {res.status_code}")
    except Exception as e:
        logger.error(f"❌ Failed to send action to backend: {e}")

def fetch_recent_metrics(limit=5):
    try:
        res = requests.get(f"http://backend:8000/metrics/recent?limit={limit}", timeout=5)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        logger.error(f"❌ Failed to fetch metrics from backend: {e}")
        return []

def build_prompt(metrics):
    return f"""
You are an AI system administrator.

## Important:
All values for CPU and memory are in **percentage (%)**.

- `"cpu": 0.9` means 0.9% — this is *low*
- `"cpu": 85.0` means 85% — this is *high*
- Only treat CPU > 70.0 as "high"
- Only treat Memory > 80.0 as "high"

Each process has:
- `name`: process name (e.g., python3)
- `cpu`: CPU usage in **percent**
- `mem`: Memory usage in **percent**
- `type`: one of `system_process`, `user_process`, `background_service`, `malware_scan`

## Rules:
- ❌ Never kill or restart `system_process`
- ✅ Kill `user_process` if CPU > 70% **or** Mem > 80%
- 🔁 Restart `background_service` if CPU > 85% **or** Mem > 85%
- 🕵️ Allow `malware_scan` unless CPU > 90% **or** Mem > 90%
- ✅ Otherwise: respond with `"none"`

# Example process:
```json
{{
  "name": "python3",
  "cpu": 0.9,
  "mem": 30.0,
  "type": "user_process"
}}

Respond in **JSON**:
{{
  "action": "none" | "kill_process" | "restart_service",
  "target": "process name or service name",
  "reason": "Why this action is needed"
}}

Metrics:
{json.dumps(metrics, indent=2)}
"""

def call_openai_agent(metrics):
    prompt = build_prompt(metrics)
    messages: list[ChatCompletionUserMessageParam] = [
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.3
    )
    content = response.choices[0].message.content
    try:
        return json.loads(content)
    except Exception as e:
        logger.error(f"Invalid OpenAI response: {content}")
        return {"action": "none"}

def call_local_ai_agent(metrics):
    logger.info("Metrics sento to build prompt: "+ str(metrics))
    prompt = build_prompt(metrics)
    logger.info("Prompt sent to ")
    try:
        res = requests.post("http://host.docker.internal:11434/api/generate", json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        })
        content = res.json().get("response", "")
        # logger.info(f"🧪 Raw Ollama response: {content}")

        # Extract JSON block
        match = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', content)
        if not match:
            match = re.search(r'({[\s\S]*?})', content)
        if not match:
            logger.error("❌ No valid JSON found in Ollama response: "+ str(content))
            return {"action": "none"}

        try:
            json_str = match.group(1)
            decision = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("🛑 Trying to fix malformed JSON...")
            try:
                import ast
                decision = ast.literal_eval(json_str)
                logger.warning("✅ Fixed using ast.literal_eval")
            except Exception as e:
                logger.error(f"❌ JSON decode failed: {e}")
                return {"action":"none"}

        # Optional: extract explanation before JSON
        explanation = content.split(json_str)[0].strip()
        if explanation:
            logger.info(f"💬 AI explanation:\n{explanation}")
            decision["explanation"] = explanation  # append explanation to use later

        return decision
    except Exception as e:
        logger.error(f"❌ Failed Ollama response: {e}")
        return {"action": "none"}

def run():
    while True:
        metrics = fetch_recent_metrics()
        if not metrics:
            logger.warning("⚠️ No metrics available, skipping analysis")
            time.sleep(30)
            continue

        host_id = metrics[0]["host_id"]

        if USE_LOCAL_AI:
            decision = call_local_ai_agent(metrics)
        else:
            decision = call_openai_agent(metrics)

        if decision["action"] != "none":
            logger.warning(f"🚨 ALERT: {decision['action']} -> {decision['target']} | Reason: {decision['reason']}")
            send_to_agent(decision, host_id=host_id)
        else:
            logger.info("✅ No action needed. System is healthy")

        time.sleep(30)

if __name__ == "__main__":
    run()
