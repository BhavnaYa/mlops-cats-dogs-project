import requests
import sys
import time

# wait for container to start
time.sleep(5)

try:
    health = requests.get("http://localhost:8000/health")
    if health.status_code != 200:
        sys.exit("Health endpoint failed")

    print("Smoke test passed")

except Exception as e:
    sys.exit(f"Smoke test failed: {e}")
