"""InferFlow için kuyruk tabanlı, çok worker'lı soket sunucusu.

Protokol: newline ile ayrılmış JSON.
  İstek:  {"input": [...]}\n
  Yanıt:  {"prediction": [...]}\n  ya da  {"error": "..."}\n

Mimari: accept() döngüsü gelen bağlantıları yerel bir kuyruğa (Queue)
bırakır; sabit sayıda "soket worker" thread'i bu kuyruktan bağlantı
çekip isteği okur. Soket nesnesi (conn) başka bir sürece aktarılamadığı
(serileştirilemediği) için, soket yönetimi hep bu process'te kalır —
ama asıl tahmin işi artık burada yapılmıyor. Soket worker'ı işi Redis
kuyruğuna (LPUSH) yazar ve sonucu benzersiz bir Redis anahtarından
(BRPOP) bekler; ayrı "tahmin worker" thread'leri de aynı kuyruktan
(BRPOP) iş çekip predict() çalıştırır ve sonucu o anahtara yazar
(RPUSH). Böylece Redis, socket'i değil, sadece veriyi taşıyan gerçek
bir ara katman olur.
"""

import json
import os
import socket
import threading
import time
import uuid
from queue import Queue

import redis

import dashboard
from model_runner import predict

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", 9000))
NUM_SOCKET_WORKERS = 4
NUM_PREDICTION_WORKERS = 4
CONNECTION_TIMEOUT_S = 30  # veri göndermeyen bir istemci worker'ı sonsuza kadar kilitlemesin

DASHBOARD_HOST = os.environ.get("DASHBOARD_HOST", HOST)
DASHBOARD_PORT = int(os.environ.get("DASHBOARD_PORT", 8080))

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
JOB_QUEUE_KEY = "inferflow:jobs"
RESULT_KEY_PREFIX = "inferflow:result:"
WORKER_KEY_PREFIX = "inferflow:worker:"
RESULT_TIMEOUT_S = 10  # tahmin worker'ı hiç cevap vermezse istemciyi sonsuza kadar bekletmemek için
RESULT_KEY_TTL_S = 30  # istemci timeout olup bağlantıyı kapattıysa, geç gelen sonuç Redis'te sonsuza kalmasın
MAX_QUEUE_DEPTH = 100  # worker'lar predict()'e yetişemiyorsa kuyruk sınırsız büyümesin, sistem "kör" olmasın

SERVER_START_TIME = time.time()  # dashboard'da uptime hesaplamak için

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True,
    # redis-py'de socket_timeout varsayılanı 5sn — BRPOP'u bundan daha
    # uzun bir süre (RESULT_TIMEOUT_S) bloklatmak istediğimizde, sunucu
    # henüz nil dönmeden istemcinin kendi soket okuması zaman aşımına
    # uğrayıp sahte bir TimeoutError fırlatıyordu. BRPOP'un kendi
    # timeout parametresi zaten bloklama süresini sınırlıyor, o yüzden
    # soket seviyesindeki timeout'u kapatmak güvenli.
    socket_timeout=None,
)

request_queue: Queue = Queue()


def recv_line(conn: socket.socket) -> bytes | None:
    """conn'dan bir newline'a kadar oku. TCP veriyi rastgele parçalar
    hâlinde teslim eder, o yüzden mesaj sınırımıza (\n) ulaşana kadar
    buffer'lıyoruz."""
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = conn.recv(4096)
        if not chunk:
            return None  # karşı taraf bağlantıyı kapattı
        buf += chunk
    return buf


def run_prediction(features) -> list:
    """İşi Redis kuyruğuna yazar, tahmin worker'ının sonucu yazmasını
    bekler. request_id her çağrıda yeniden üretiliyor çünkü kuyruk tek
    ama aynı anda bekleyen istemci birden fazla olabilir — sonuç,
    isteği gönderenden başkasına gitmemeli."""
    request_id = uuid.uuid4().hex
    job = json.dumps({"id": request_id, "input": features})
    redis_client.lpush(JOB_QUEUE_KEY, job)

    result_key = RESULT_KEY_PREFIX + request_id
    popped = redis_client.brpop(result_key, timeout=RESULT_TIMEOUT_S)
    if popped is None:
        raise TimeoutError("tahmin worker'ından zamanında yanıt gelmedi")

    _, raw_result = popped
    return json.loads(raw_result)


def process_connection(worker_id: int, conn: socket.socket, addr) -> None:
    with conn:
        # İstemci bağlanıp hiç veri göndermezse (ya da yarım bırakırsa)
        # recv() sonsuza kadar bloklardı ve worker'ı kalıcı olarak
        # kilitlerdi — 30sn sonra pes edip bağlantıyı temiz kapatıyoruz.
        conn.settimeout(CONNECTION_TIMEOUT_S)
        try:
            line = recv_line(conn)
        except socket.timeout:
            print(f"[socket-worker-{worker_id}] {addr} -> {CONNECTION_TIMEOUT_S}s içinde veri gelmedi, bağlantı kapatılıyor")
            return
        if line is None:
            return

        start = time.perf_counter()
        try:
            payload = json.loads(line)
            queue_depth = redis_client.llen(JOB_QUEUE_KEY)
            if queue_depth >= MAX_QUEUE_DEPTH:
                # İşi Redis'e (LPUSH) yazmadan önce bakıyoruz: worker'lar
                # yetişemiyorsa kuyruğu daha da şişirmek yerine isteği
                # baştan reddediyoruz — backpressure budur.
                response = {"error": "server busy, try again later"}
                print(f"[socket-worker-{worker_id}] {addr} -> queue_full: rejected due to backpressure (depth={queue_depth})")
            else:
                response = {"prediction": run_prediction(payload["input"])}
        except Exception as exc:  # bozuk girdide ya da worker gecikmesinde soket worker'ı ayakta tut
            response = {"error": str(exc)}
        elapsed_ms = (time.perf_counter() - start) * 1000

        conn.sendall((json.dumps(response) + "\n").encode())
        # worker_id logda görünsün ki isteklerin worker'lar arasında
        # nasıl dağıldığı (adil mi, tek worker mı boğuluyor) izlenebilsin.
        print(f"[socket-worker-{worker_id}] {addr} -> handled in {elapsed_ms:.1f}ms | queue depth: {request_queue.qsize()}")


def socket_worker_loop(worker_id: int) -> None:
    # Her worker sonsuz döngüde kuyruktan iş çeker. Kuyruk boşsa
    # get() otomatik olarak yeni iş gelene kadar bloklar (busy-wait
    # yapmayız, CPU boşa harcanmaz).
    while True:
        conn, addr = request_queue.get()
        process_connection(worker_id, conn, addr)
        request_queue.task_done()


def prediction_worker_loop(worker_id: int) -> None:
    # Worker durumunu Redis'e yazıyoruz ki dashboard, bu process'in
    # dışından (ayrı bir HTTP sunucusundan) hangi worker'ın meşgul/boşta
    # olduğunu ve kaç iş işlediğini görebilsin.
    worker_key = WORKER_KEY_PREFIX + str(worker_id)
    total_handled = 0
    redis_client.hset(
        worker_key,
        mapping={"status": "idle", "total_handled": total_handled, "last_job": "-", "last_heartbeat": time.time()},
    )

    # timeout=0 (süresiz blok) yerine sonlu bir timeout ile bekliyoruz:
    # redis-py istemcisi bazı sürümlerde süresiz bloklarken soket
    # okuma timeout'una takılıp thread'i çökertebiliyor. Sonlu timeout
    # + boşsa döngüye devam etmek busy-wait yaratmadan bu riski ortadan
    # kaldırıyor.
    while True:
        popped = redis_client.brpop(JOB_QUEUE_KEY, timeout=5)
        # İş gelmese bile heartbeat'i tazeliyoruz: dashboard bu sayede
        # "worker boşta bekliyor" ile "worker thread'i çökmüş, bir daha
        # hiç güncellenmedi" durumunu ayırt edebiliyor.
        redis_client.hset(worker_key, "last_heartbeat", time.time())
        if popped is None:
            continue
        _, raw_job = popped
        job = json.loads(raw_job)

        redis_client.hset(worker_key, mapping={"status": "busy", "last_job": job["id"]})
        result = predict(job["input"])

        result_key = RESULT_KEY_PREFIX + job["id"]
        redis_client.rpush(result_key, json.dumps(result))
        redis_client.expire(result_key, RESULT_KEY_TTL_S)

        total_handled += 1
        redis_client.hset(worker_key, mapping={"status": "idle", "total_handled": total_handled, "last_heartbeat": time.time()})
        print(f"[prediction-worker-{worker_id}] job {job['id']} handled")


def main() -> None:
    for worker_id in range(NUM_SOCKET_WORKERS):
        threading.Thread(target=socket_worker_loop, args=(worker_id,), daemon=True).start()

    for worker_id in range(NUM_PREDICTION_WORKERS):
        threading.Thread(target=prediction_worker_loop, args=(worker_id,), daemon=True).start()

    threading.Thread(
        target=dashboard.run,
        args=(
            redis_client,
            JOB_QUEUE_KEY,
            WORKER_KEY_PREFIX,
            NUM_PREDICTION_WORKERS,
            MAX_QUEUE_DEPTH,
            SERVER_START_TIME,
            DASHBOARD_HOST,
            DASHBOARD_PORT,
        ),
        daemon=True,
    ).start()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((HOST, PORT))
        srv.listen(16)
        print(
            f"InferFlow listening on {HOST}:{PORT} "
            f"(socket_workers={NUM_SOCKET_WORKERS}, prediction_workers={NUM_PREDICTION_WORKERS})"
        )

        while True:
            conn, addr = srv.accept()
            # accept() işi hemen kuyruğa bırakır, worker'ları beklemez —
            # bu sayede sunucu ne kadar iş birikirse biriksin bağlantı
            # kabul etmeye devam eder.
            request_queue.put((conn, addr))


if __name__ == "__main__":
    main()
