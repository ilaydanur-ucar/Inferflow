"""InferFlow icin Locust yuk testi.

InferFlow HTTP degil, newline-delimited JSON konusan ham bir TCP
sunucusu — Locust'un HTTP istemcisi burada ise yaramaz. Bunun yerine
Locust'un dokumante ettigi "custom client" deseniyle kendi TCP
istemcimizi yaziyoruz ve olcumu elle events.request.fire() ile
Locust'a bildiriyoruz.

Calistirma (host'tan, docker compose zaten ayaktayken):
    pip install -r requirements-dev.txt
    locust -f locustfile.py --host tcp://127.0.0.1:9000
Web arayuzu icin http://localhost:8089 acilir; headless icin:
    locust -f locustfile.py --headless -u 50 -r 10 -t 30s
"""

import json
import os
import socket
import time

from locust import User, task, between, events

HOST = os.environ.get("INFERFLOW_HOST", "127.0.0.1")
PORT = int(os.environ.get("INFERFLOW_PORT", 9000))

SAMPLE_INPUT = [5.1, 3.5, 1.4, 0.2]


class InferFlowClient:
    def predict(self, features):
        start = time.perf_counter()
        exception = None
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((HOST, PORT))
                s.sendall((json.dumps({"input": features}) + "\n").encode())

                buf = b""
                while not buf.endswith(b"\n"):
                    chunk = s.recv(4096)
                    if not chunk:
                        raise ConnectionError("sunucu baglantiyi erken kapatti")
                    buf += chunk

                response = json.loads(buf)
                if "error" in response:
                    raise RuntimeError(response["error"])
        except Exception as exc:
            exception = exc

        elapsed_ms = (time.perf_counter() - start) * 1000
        # Locust'un HTTP-disi protokolleri de raporlayabilmesi icin
        # olcumu elle bildiriyoruz (request_type="TCP", HTTP degil).
        events.request.fire(
            request_type="TCP",
            name="predict",
            response_time=elapsed_ms,
            response_length=0,
            exception=exception,
        )


class InferFlowUser(User):
    wait_time = between(0, 0.05)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = InferFlowClient()

    @task
    def predict(self):
        self.client.predict(SAMPLE_INPUT)
