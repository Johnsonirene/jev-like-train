"""Start a real checkpoint, exercise HTTP, then stop the test server."""
import argparse
import base64
from io import BytesIO
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, ProxyHandler

parser = argparse.ArgumentParser()
parser.add_argument("service", choices=["autojev", "agentjev"])
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--backbone")
parser.add_argument("--output", required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
env["PYTHONPATH"] = str(root / "src")
key = "local-smoke-test-key"
opener = build_opener(ProxyHandler({}))
if args.service == "autojev":
    env.update(AUTOJEV_CHECKPOINT=str(Path(args.checkpoint).resolve()), AUTOJEV_API_KEY=key, AUTOJEV_HOST="127.0.0.1", PORT=str(port))
    cmd = [sys.executable, "-m", "autojev.server"]
else:
    if not args.backbone:
        parser.error("agentjev requires --backbone")
    cmd = [sys.executable, str(root / "agentjev_api/scripts/serve.py"), "--run-dir", args.checkpoint, "--backbone", args.backbone, "--port", str(port)]

def call(path, payload=None, authenticated=True):
    headers = {"Content-Type": "application/json"}
    if authenticated:
        headers["Authorization"] = "Bearer " + key
    request = Request(f"http://127.0.0.1:{port}" + path, data=None if payload is None else json.dumps(payload).encode(), headers=headers)
    try:
        with opener.open(request, timeout=60) as response:
            data = response.read()
            return response.status, json.loads(data) if "application/json" in response.headers.get("Content-Type", "") else data.decode()
    except HTTPError as e:
        return e.code, json.loads(e.read())

result = {"service": args.service, "checkpoint": str(Path(args.checkpoint).resolve()), "offline": True, "checks": {}}
start = time.monotonic()
with tempfile.TemporaryFile(mode="w+") as log:
    process = subprocess.Popen(cmd, cwd=root, env=env, stdout=log, stderr=log)
    try:
        deadline = time.monotonic() + 120
        while True:
            if process.poll() is not None:
                raise RuntimeError("Server exited before becoming ready")
            try:
                status, health = call("/health")
                if status == 200 and health["status"] == "ready":
                    break
            except (URLError, TimeoutError, ConnectionError):
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("Server startup exceeded 120 seconds")
            time.sleep(0.5)
        result["health"] = health
        result["checks"]["health"] = True
        assert call("/")[0] == 200
        result["checks"]["root"] = True
        if args.service == "autojev":
            endpoint = "/v1/systemone"
            payload = {"model": "jev-latest", "state": "The customer was charged twice for one order and asks for a refund.", "questions": {
                "route": {"type": "choice", "instructions": "Which department should handle the request?", "criteria": {"billing": "Duplicate charge or refund", "tech": "Software problem"}},
                "refund": {"type": "noul", "instructions": "Is the customer requesting a refund?"},
                "urgency": {"type": "score", "instructions": "Rate urgency.", "criteria": ["Low", "Medium", "High"]}}}
            assert call(endpoint, payload, authenticated=False)[0] == 401
            assert call("/v1/models")[0] == call("/docs")[0] == 200
            result["checks"]["authentication_and_docs"] = True
        else:
            endpoint = "/api/evaluate"
            payload = {"id": "synthetic-smoke", "state": "The customer was charged twice for one order and asks for a refund.", "questions": [
                {"id": "route", "type": "choice", "question": "Which department should handle the request?", "options": {"billing": "Duplicate charge or refund", "tech": "Software problem"}},
                {"id": "refund", "type": "boolean", "question": "Is the customer requesting a refund?"},
                {"id": "urgency", "type": "score", "question": "Rate urgency.", "levels": ["Low", "Medium", "High"]}]}
        status, response = call(endpoint, payload)
        assert status == 200, response
        result["response"] = response
        answers = list(response["answers"].values()) if args.service == "autojev" else response["results"][0]["answers"]
        assert len(answers) == 3
        for answer in answers:
            if answer["type"] == "noul":
                assert math.isfinite(answer["noul"]) and 0 <= answer["noul"] <= 1
            else:
                values = list(answer.get("probabilities", answer.get("distribution")).values())
                assert all(math.isfinite(x) and 0 <= x <= 1 for x in values)
                assert abs(sum(values) - 1) < 1e-4
        assert response["usage"].get("output_tokens", response["usage"].get("generated_tokens")) == 0
        result["checks"]["three_primitives_and_probabilities"] = True
        invalid = json.loads(json.dumps(payload))
        if args.service == "autojev":
            invalid["questions"]["route"]["criteria"] = {}
        else:
            invalid["questions"][0]["options"] = []
        assert call(endpoint, invalid)[0] == (422 if args.service == "autojev" else 400)
        result["checks"]["invalid_request"] = True
        if args.service == "autojev":
            from PIL import Image
            image = BytesIO()
            Image.new("RGB", (64, 64), "red").save(image, format="PNG")
            image_payload = {"model": "jev-latest", "state": "Look at the image.", "questions": {"color": {"type": "choice", "instructions": "What is the dominant color?", "criteria": {"red": "Red", "blue": "Blue"}}}, "images": ["data:image/png;base64," + base64.b64encode(image.getvalue()).decode()]}
            status, image_response = call(endpoint, image_payload)
            assert status == 200, image_response
            probabilities = image_response["answers"]["color"]["probabilities"]
            assert abs(sum(probabilities.values()) - 1) < 1e-4
            result["image_response"] = image_response
            result["checks"]["image_inference"] = True
        result["passed"] = True
    except Exception:
        process.terminate()
        process.wait(timeout=15)
        log.seek(0)
        print(log.read()[-12000:], file=sys.stderr)
        raise
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
result["elapsed_seconds"] = round(time.monotonic() - start, 3)
Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"service": args.service, "passed": result["passed"], "checks": result["checks"], "elapsed_seconds": result["elapsed_seconds"]}))
