"""InferFlow için basit test istemcisi.

Sunucuya bağlanır, girdi gönderir, yanıtı ve gecikmeyi (latency) yazdırır.
"""

import json
import socket
import time

HOST, PORT = "127.0.0.1", 9000


def predict(features: list[float]) -> dict:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))

        start = time.perf_counter()
        s.sendall((json.dumps({"input": features}) + "\n").encode())

        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(4096)
            if not chunk:
                break
            buf += chunk
        elapsed_ms = (time.perf_counter() - start) * 1000

        result = json.loads(buf)
        result["_client_latency_ms"] = round(elapsed_ms, 2)
        return result


if __name__ == "__main__":
    ornek = [5.1, 3.5, 1.4, 0.2]
    print(predict(ornek))
