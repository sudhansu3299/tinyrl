import os

import time
import requests
import psutil
from datetime import datetime
from config import BACKEND_URL, HOST_ID, INTERVAL
import sys

import logging
import threading

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)

logger = logging.getLogger(__name__)

def simulate_memory_spike(chunk_size_mb=2):
    logger.info("🚨 Starting memory spike simulation")
    memory_hog = []
    for i in range(300):
        memory_hog.append(bytearray(chunk_size_mb * 1024 * 1024))  # Allocate real memory
        logger.info(f"📈 Allocated {(i+1)*chunk_size_mb} MB")
        time.sleep(1)
    logger.info("Finished memory spike simulation!")

def get_container_memory_percent():
    try:
        with open('/sys/fs/cgroup/memory.current') as f:
            usage = int(f.read())
        with open('/sys/fs/cgroup/memory.max') as f:
            limit_str = f.read().strip()
            if limit_str == "max":
                return psutil.virtual_memory().percent  # fallback to host-level if no limit
            limit = int(limit_str)
        if limit == 0:
            return 0.0
        return round((usage / limit) * 100, 2)
    except Exception as e:
        logger.warning(f"⚠️ Failed to read cgroup v2 memory stats: {e}")
        return psutil.virtual_memory().percent  # fallback

def get_container_cpu_percent(interval=1.0):
    try:
        cpu_stat_path = "/sys/fs/cgroup/cpu.stat"
        with open(cpu_stat_path, "r") as f:
            lines = f.readlines()
            usage_usec_line = next((line for line in lines if line.startswith("usage_usec")), None)
            if usage_usec_line:
                usage_usec_1 = int(usage_usec_line.strip().split()[1])
            else:
                raise Exception("usage_usec not found")

        time.sleep(interval)

        with open(cpu_stat_path, "r") as f:
            lines = f.readlines()
            usage_usec_line = next((line for line in lines if line.startswith("usage_usec")), None)
            if usage_usec_line:
                usage_usec_2 = int(usage_usec_line.strip().split()[1])
            else:
                raise Exception("usage_usec not found")

        delta_usec = usage_usec_2 - usage_usec_1
        num_cores = os.cpu_count()
        # Convert microseconds to percentage
        return round((delta_usec / (interval * 1e6 * num_cores)) * 100, 2)

    except Exception as e:
        logger.warning(f"⚠️ Failed to read cgroup v2 CPU stats: {e}")
        return psutil.cpu_percent(interval=interval)  # fallback

def collect_metrics():
    cpu = get_container_cpu_percent()
    mem = get_container_memory_percent()
    disk = psutil.disk_usage('/').percent

    processes = sorted(
        [(p.pid, p.name(), p.cpu_percent(interval=None)) for p in psutil.process_iter(['pid', 'name'])],
        key=lambda x: x[2], reverse=True
    )[:5]

    top_processes = [{"pid": pid, "name": name, "cpu": round(cpu_val, 2)} for pid, name, cpu_val in processes]

    data =  {
        "timestamp": datetime.utcnow().isoformat(),
        "host_id": HOST_ID,
        "cpu_percent": cpu,
        "memory_percent": mem,
        "disk_percent": disk,
        "top_processes": top_processes
    }

    logger.info(f"📡 Collected metrics: CPU={cpu}%, MEM={mem}%, DISK={disk}%")
    return data

def send_metrics(data):
    try:
        response = requests.post(f"{BACKEND_URL}/metrics", json=data, timeout=5)
        logger.info(f"[{data['timestamp']}] ✅ Sent metrics: {response.status_code}")
    except Exception as e:
        logger.error(f"❌ Failed to send metrics: {e}")

def run_agent():
    while True:
        metrics = collect_metrics()
        send_metrics(metrics)
        time.sleep(INTERVAL)

if __name__ == "__main__":
    # threading.Thread(target=simulate_memory_spike, daemon=True).start()
    run_agent()
