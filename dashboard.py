"""InferFlow için canlı izleme paneli.

Backend tarafında burada da bir web framework'ü kullanmıyoruz:
stdlib'deki http.server ile tek bir /stats JSON endpoint'i sunuyoruz.
Görsel panonun kendisi artık ayrı bir React/TanStack Start uygulaması
(frontend/) — bu dosya onu derlenmiş static dosyalar (frontend/dist/client)
olarak aynı porttan servis ediyor, böylece prod'da tek process/tek port
yeterli oluyor ve Node sunucusuna hiç ihtiyaç kalmıyor.

Redis burada "gerçek" veri kaynağı — kuyruk derinliği LLEN ile, worker
durumları prediction_worker_loop'un yazdığı hash'lerden okunuyor.
Health-check göstergeleri (Redis bağlı mı, worker'lar canlı mı) da
dekoratif değil: Redis'e gerçek bir PING atılıyor, worker'ların
canlılığı da her BRPOP turunda tazelenen bir "last_heartbeat"
zaman damgasının ne kadar eskidiği ölçülerek belirleniyor — bir worker
thread'i çökerse heartbeat'i bir daha güncellenmez, dashboard bunu
"offline" olarak gösterir.
"""

import json
import mimetypes
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HEARTBEAT_INTERVAL_S = 5  # prediction_worker_loop'un BRPOP timeout'uyla aynı
ALIVE_THRESHOLD_S = 3 * HEARTBEAT_INTERVAL_S  # birkaç turu kaçırmak jitter olabilir, kesin çökme değil

# React dashboard'unun `npm run build` (frontend/) çıktısı. Bu dizin yoksa
# (frontend henüz build edilmediyse) statik dosya servisi 503 döner.
FRONTEND_DIST = (Path(__file__).parent / "frontend" / "dist" / "client").resolve()


def _collect_stats(redis_client, job_queue_key, worker_key_prefix, num_workers, max_queue_depth, server_start_time):
    try:
        redis_connected = bool(redis_client.ping())
    except Exception:
        redis_connected = False

    queue_depth = redis_client.llen(job_queue_key) if redis_connected else 0
    now = time.time()

    workers = []
    total_handled = 0
    for worker_id in range(num_workers):
        data = redis_client.hgetall(worker_key_prefix + str(worker_id)) if redis_connected else {}
        heartbeat = data.get("last_heartbeat")
        alive = heartbeat is not None and (now - float(heartbeat)) < ALIVE_THRESHOLD_S
        handled = int(data.get("total_handled", 0))
        total_handled += handled
        workers.append({
            "id": worker_id,
            "status": data.get("status", "unknown"),
            "total_handled": handled,
            "last_job_id": data.get("last_job", "-"),
            "alive": alive,
        })

    return {
        "queue_depth": queue_depth,
        "max_queue_depth": max_queue_depth,
        "workers": workers,
        "total_handled": total_handled,
        "redis_connected": redis_connected,
        "uptime_seconds": int(now - server_start_time),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }


def _render_prometheus(stats):
    """_collect_stats()'ın çıktısını Prometheus text exposition formatına
    çevirir, böylece bir Prometheus sunucusu bu paneli scrape edebilir."""
    lines = [
        "# HELP inferflow_queue_depth Current depth of the Redis job queue.",
        "# TYPE inferflow_queue_depth gauge",
        f"inferflow_queue_depth {stats['queue_depth']}",
        "# HELP inferflow_queue_max_depth Configured max queue depth before backpressure kicks in.",
        "# TYPE inferflow_queue_max_depth gauge",
        f"inferflow_queue_max_depth {stats['max_queue_depth']}",
        "# HELP inferflow_redis_connected Whether the dashboard's Redis PING succeeded (1) or not (0).",
        "# TYPE inferflow_redis_connected gauge",
        f"inferflow_redis_connected {1 if stats['redis_connected'] else 0}",
        "# HELP inferflow_uptime_seconds Seconds since the server process started.",
        "# TYPE inferflow_uptime_seconds counter",
        f"inferflow_uptime_seconds {stats['uptime_seconds']}",
        "# HELP inferflow_worker_total_handled Total prediction jobs handled by a worker since start.",
        "# TYPE inferflow_worker_total_handled counter",
    ]
    for w in stats["workers"]:
        lines.append(f'inferflow_worker_total_handled{{worker_id="{w["id"]}"}} {w["total_handled"]}')
    lines.append("# HELP inferflow_worker_alive Whether a worker's heartbeat is fresh (1) or stale/crashed (0).")
    lines.append("# TYPE inferflow_worker_alive gauge")
    for w in stats["workers"]:
        lines.append(f'inferflow_worker_alive{{worker_id="{w["id"]}"}} {1 if w["alive"] else 0}')
    lines.append("# HELP inferflow_worker_busy Whether a worker is currently processing a job (1) or idle (0).")
    lines.append("# TYPE inferflow_worker_busy gauge")
    for w in stats["workers"]:
        lines.append(f'inferflow_worker_busy{{worker_id="{w["id"]}"}} {1 if w["status"] == "busy" else 0}')
    return ("\n".join(lines) + "\n").encode()


def _resolve_static_path(url_path):
    """`/`, `/assets/index-abc.js` gibi bir istek yolunu frontend/dist/client
    içindeki gerçek bir dosyaya çevirir. Path traversal'a (`/../../secrets`)
    karşı, çözülen yolun FRONTEND_DIST'in dışına çıkmadığını doğruluyoruz.
    Dosya adında nokta yoksa (örn. `/` gibi client-route benzeri bir yol)
    SPA'nın tek giriş noktası olan index.html'e düşüyoruz."""
    clean_path = url_path.split("?", 1)[0]
    relative = clean_path.lstrip("/") or "index.html"
    candidate = (FRONTEND_DIST / relative).resolve()
    if candidate == FRONTEND_DIST or FRONTEND_DIST not in candidate.parents:
        return None  # traversal denemesi
    if not candidate.is_file():
        if "." not in candidate.name:
            candidate = FRONTEND_DIST / "index.html"
        if not candidate.is_file():
            return None
    return candidate


def _make_handler(redis_client, job_queue_key, worker_key_prefix, num_workers, max_queue_depth, server_start_time):
    class StatsHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # her fetch() istegi ana loglari bogmasin diye erisim loglarini kapatiyoruz

        def do_GET(self):
            if self.path == "/stats":
                body = json.dumps(
                    _collect_stats(redis_client, job_queue_key, worker_key_prefix, num_workers, max_queue_depth, server_start_time)
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path == "/metrics":
                stats = _collect_stats(redis_client, job_queue_key, worker_key_prefix, num_workers, max_queue_depth, server_start_time)
                body = _render_prometheus(stats)
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if not FRONTEND_DIST.is_dir():
                body = (
                    b"Frontend henuz build edilmedi. `cd frontend && npm install && npm run build` "
                    b"calistirip tekrar deneyin."
                )
                self.send_response(503)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            static_path = _resolve_static_path(self.path)
            if static_path is None:
                self.send_response(404)
                self.end_headers()
                return

            content_type, _ = mimetypes.guess_type(static_path.name)
            body = static_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type or "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return StatsHandler


def run(redis_client, job_queue_key, worker_key_prefix, num_workers, max_queue_depth, server_start_time, host="0.0.0.0", port=8080):
    handler = _make_handler(redis_client, job_queue_key, worker_key_prefix, num_workers, max_queue_depth, server_start_time)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Dashboard listening on {host}:{port}")
    server.serve_forever()
