# InferFlow

Framework kullanmadan (Flask, FastAPI, gRPC yok) ham TCP socket'ler üzerine yazılmış bir ML model inference servisi. Bir istemci newline ile ayrılmış bir JSON isteği gönderiyor, sunucu bunu bir Redis kuyruğu üzerinden bir worker havuzuna dağıtıyor, worker bir ONNX modeliyle (scikit-learn RandomForest, Iris veri seti) tahmin üretip cevabı geri döndürüyor.

Amaç production'a çıkacak bir ürün yapmak değildi. Amaç, normalde bir web framework'ünün ya da bir model-serving aracının (TensorFlow Serving, TorchServe, Triton) arkasında gizlenen şeyleri — istek kuyruğu, worker havuzu, backpressure, health check, container orkestrasyon — elle kurup gerçekten nasıl çalıştığını görmekti. "Framework'e yaslanmadan arkasında ne olduğunu anlıyorum" iddiasının arkasında durabilmek.

Bu, yaklaşık bir aylık bir öğrenme projesidir. Aşağıdaki mimari, teknik detaylar ve kısıtlar projenin bittiği andaki gerçek durumunu yansıtıyor — planlanan değil, çalışan.

## Mimari

```
TCP İstemcileri
      │
      ▼
Socket Sunucu — accept() döngüsü
      │  4 "socket worker" thread
      │  - bağlantıyı okur (30s timeout)
      │  - Redis kuyruk derinliğini kontrol eder (backpressure)
      ▼
Redis Kuyruğu — inferflow:jobs (max depth: 100)
      │  4 "tahmin worker" thread
      │  - BRPOP ile pull-based iş çeker (round-robin değil,
      │    kuyruk kendini doğal olarak dengeler)
      ▼
ONNX Model — RandomForest, CPU (onnxruntime)
      │
      ▼
Redis sonuç anahtarı — inferflow:result:<id>
      │  (socket worker bunu BRPOP ile bekliyordu)
      ▼
İstemciye yanıt
```

İki ayrı worker havuzu olması bilinçli bir sonuç: socket'i (bağlantı nesnesini) bir process'ten diğerine ya da Redis'e aktaramazsın, serileştirilemez. O yüzden soket yönetimi hep ana process'te kalıyor; sadece **veri** (JSON payload) Redis üzerinden akıyor. Bu ayrımın neden önemli olduğu Bölüm 7'de.

## Teknik detaylar

**Protokol:** newline ile ayrılmış JSON, ham TCP üzerinde.

```
İstek:  {"input": [5.1, 3.5, 1.4, 0.2]}\n
Yanıt:  {"prediction": [0]}\n
   ya da {"error": "..."}\n
```

**Worker dağılımı:** Round-robin değil, pull-based. Worker'lar boşaldıkça `BRPOP` ile kuyruktan kendileri iş çeker — kuyruk, yükü worker'lar arasında doğal olarak dengeler; hiçbir worker'a elle iş atanmıyor.

**Backpressure:** İstek geldiğinde, Redis'e yazılmadan önce `inferflow:jobs` kuyruğunun derinliği (`LLEN`) kontrol edilir. 100'e ulaşmışsa istek reddedilir (`{"error": "server busy, try again later"}`), bağlantı kapatılır.

**Health check:** `docker-compose.yml`'de Redis'e bir `healthcheck` (`redis-cli ping`) tanımlı; `app` servisi `depends_on: condition: service_healthy` ile Redis tam hazır olana kadar başlamıyor.

**Socket timeout:** Bir istemci bağlanıp hiç veri göndermezse (`conn.settimeout(30)`), worker 30 saniye sonra bağlantıyı temiz kapatıp başka işe devam ediyor — sonsuza kadar kilitlenmiyor.

**Per-IP bağlantı limiti:** Sadece 4 socket worker olduğu için, tek bir istemcinin art arda açtığı bağlantılar (kasıtlı ya da hatalı bir client) tüm havuzu kilitleyebilir. `accept()` anında, aynı IP'den `MAX_CONNECTIONS_PER_IP` (varsayılan 60, env var ile ayarlanabilir) değerini aşan bağlantılar kuyruğa hiç girmeden reddediliyor. Varsayılan yüksek tutuldu çünkü `locustfile.py` 50 eşzamanlı kullanıcıyı tek makineden (dolayısıyla tek IP'den) simüle ediyor — üretimde çok sayıda gerçek istemci varsa bu değer düşürülebilir.

**Worker crash recovery:** Tahmin worker'ları job'u `BRPOP` yerine `BLMOVE` ile çekiyor — job, ana kuyruktan silinip worker'a özel bir `inferflow:processing:<id>` listesine taşınıyor. İş bitince (başarılı ya da hatalı) oradan siliniyor. Worker ya da tüm process job'u bitirmeden çökerse, iş bu listede kalır ve bir sonraki başlangıçta otomatik olarak ana kuyruğa geri alınır — yani artık **at-least-once delivery** var (önceki at-most-once yerine); bedeli, çok nadir durumda aynı job'un iki kez işlenebilmesi.

## Kurulum ve çalıştırma

**Ön koşullar:** Docker Desktop (WSL2 entegrasyonlu), `docker compose` komutu (Docker Desktop'a dahil).

**Çalıştırma:**

```bash
docker compose up --build
```

Bu, iki container ayağa kaldırır (`redis`, `app`); `app` container'ı ayağa kalkarken önce modeli eğitip ONNX'e çevirir (`model/train_and_export.py`), sonra sunucuyu başlatır.

**Test (host'tan):**

```bash
python client.py
```

**Dashboard:** Tarayıcıda `http://localhost:8080` — canlı kuyruk derinliği ve worker durumları (idle/busy, işlenen iş sayısı). Arayüz artık `frontend/`'de ayrı bir React (TanStack Start) uygulaması; derlenmiş çıktısı (`frontend/dist/client`) `dashboard.py` tarafından aynı porttan statik dosya olarak servis ediliyor — prod'da tek process, tek port, Node sunucusu yok. İlk çalıştırmadan önce build etmeniz gerekir:

```bash
cd frontend
npm install
npm run build   # frontend/dist/client üretir, dashboard.py bunu servis eder
```

Frontend üzerinde çalışırken (hot reload için):

```bash
cd frontend
npm run dev   # http://localhost:5173, /stats isteklerini :8080'e proxy'ler
```

**Locust ile yük testi:**

```bash
pip install -r requirements-dev.txt
locust -f locustfile.py --headless -u 50 -r 10 -t 60s --host tcp://127.0.0.1:9000
```

InferFlow HTTP değil ham TCP konuştuğu için, `locustfile.py` Locust'un HTTP istemcisi yerine kendi TCP istemcisini kullanıp ölçümü `events.request.fire()` ile elle bildiriyor.

**Deploy (opsiyonel):** `inferflow.service` adında bir systemd unit dosyası hazır (Docker Compose'u `up`/`down` ile yönetiyor, çökerse yeniden başlatıyor) — henüz sisteme kurulmadı, kurulum adımları dosyanın yanındaki notlarda. Container seviyesinde de `docker-compose.yml`'de her iki servise `restart: unless-stopped` tanımlı; `app` çökerse (örn. beklenmeyen bir exception) Docker onu otomatik yeniden başlatır.

**CI:** `.github/workflows/ci.yml`, her push/PR'da `docker compose up --build` ile tüm stack'i ayağa kaldırıp gerçek bir tahmin isteği gönderiyor ve `/stats`'ın bunu doğru yansıttığını doğruluyor (build + smoke test, birim test değil).

**Metrikler:** `/stats`'a ek olarak `http://localhost:8080/metrics`, aynı veriyi Prometheus text exposition formatında sunuyor (`inferflow_queue_depth`, `inferflow_worker_total_handled{worker_id=...}`, `inferflow_worker_alive`, vb.) — bir Prometheus sunucusu doğrudan scrape edebilir.

## Benchmark sonuçları

| Metrik | Değer |
|---|---|
| Toplam İstek | 19.651 |
| Başarısız İstek | 0 (%0) |
| RPS (istek/saniye) | ~331 |
| p50 Latency | 110ms |
| p95 Latency | 230ms |
| p99 Latency | 320ms |
| Test Süresi | 60s |
| Eşzamanlı Kullanıcı | 50 (ramp-up: 10/s) |

Not: p99.9 ve üzeri birkaç istek ~1000ms'e kadar çıktı (max: 1023ms) — WSL2 üzerinde Docker Desktop'ın network forwarding katmanından kaynaklanan ara sıra bir gecikme sıçraması olduğunu düşünüyoruz, ama kesin sebep araştırılmadı; ana kütlenin (%99'u) 320ms altında kaldığını not ediyoruz.

## Kullanılan teknolojiler

- **Dil:** Python 3.11
- **Model:** scikit-learn RandomForest → ONNX (onnxruntime, CPU)
- **Ağ:** Ham TCP socket, newline-delimited JSON
- **Eşzamanlılık:** Python threading + Redis kuyruğu
- **Kuyruk:** Redis 7 (Alpine)
- **Container:** Docker + Docker Compose
- **Deploy:** systemd unit dosyası (`inferflow.service`, hazır)
- **Yük testi:** Locust
- **İzleme:** Backend `/stats` JSON ve `/metrics` (Prometheus text format) endpoint'leri framework'süz (stdlib `http.server`); arayüz `frontend/`'de React 19 + TanStack Start + Tailwind CSS v4 ile yazılı, `fetch` ile saniyede bir güncelleniyor, `/stats`'a ulaşamazsa yerel bir simülasyona düşüp "SIM" rozetini gösteriyor
- **CI:** GitHub Actions (`.github/workflows/ci.yml`) — her push/PR'da build + uçtan uca smoke test

## Mimarinin gerçek darboğazı (öğrenilen ders)

Backpressure kodu doğru çalışıyor, ama bu projede Redis kuyruğunu **gerçek yükle** doldurmak neredeyse imkansız — `predict()` (küçük bir RandomForest, tek örnek) sub-milisaniye sürüyor, 4 tahmin worker'ı bunu anında tüketiyor.

Asıl darboğaz Redis değil, `NUM_SOCKET_WORKERS=4`: aynı anda en fazla 4 TCP bağlantısı okunup Redis'e iş olarak yazılabiliyor. 50 eşzamanlı Locust kullanıcısıyla test edildiğinde, 46'sı Redis'e hiç ulaşmadan yerel `request_queue`'da (Python'ın kendi `queue.Queue`'su) bekliyor. Backpressure kod yolunu gerçekten tetiklemek için Redis'e doğrudan (bir Lua script ile, atomik olarak) 50.000 sahte iş yazıp yapay bir backlog oluşturmak gerekti — sadece o zaman gerçek bir istek `"server busy"` yanıtı aldı.

Bunun anlamı: bu spesifik sistemde darboğaz kuyrukta değil, kuyruğa girişte. Model çok daha ağır olsaydı (örneğin bir LLM inference'ı, network round-trip'i olan bir dış servis çağrısı) ya da `NUM_SOCKET_WORKERS` sayısı artırılsaydı, backpressure gerçek koşullarda da devreye girerdi. Sistemin nerede tıkanacağını bilmek, "her yere kuyruk koydum" demekten daha değerli.

## Karşılaşılan zorluklar ve çözümleri

- **`HOST=127.0.0.1` container içinde dışarıdan erişilemiyordu** — Docker'ın port yönlendirmesi container'ın loopback'ine değil, kendi network arayüzüne ulaşıyor. Çözüm: `HOST` env var'a bağlandı, Dockerfile'da `ENV HOST=0.0.0.0`.
- **`docker compose up` sonrası log hiç görünmüyordu** — Python, TTY olmayan bir ortamda stdout'u buffer'lıyor; sunucu hiç çıkmadığı için buffer flush edilmiyordu. Çözüm: `ENV PYTHONUNBUFFERED=1`.
- **Redis-backed worker'lar birkaç saniye içinde sessizce çöküyordu** — redis-py 8.x'te istemcinin varsayılan `socket_timeout`'u 5 saniye (önceki sürümlerde sınırsızdı). `BRPOP` sunucu tarafında 5-10 saniyeye kadar bloklayabiliyordu; istemci kendi soket okumasında daha erken pes edip sahte bir `TimeoutError` fırlatıyordu. Kök nedeni, container içine girip Redis client'ın gerçek `connection_kwargs`'ını inceleyerek bulduk. Çözüm: `socket_timeout=None` — bloklama süresini zaten `BRPOP`'un kendi `timeout` parametresi sınırlıyor.
- **İstemci bağlanıp veri göndermezse worker sonsuza kadar kilitleniyordu** — Çözüm: `conn.settimeout(30)`, `socket.timeout` yakalanıp bağlantı temiz kapatılıyor.
- **Redis hazır olmadan app başlıyor, ilk isteklerde bağlantı hatası veriyordu** — Çözüm: `docker-compose.yml`'de Redis'e `healthcheck`, app'e `depends_on: condition: service_healthy`.

## Bilinen kısıtlar ve kapsam dışı bırakılanlar

Bunların hiçbiri "unutuldu" değil — zaman ve kapsam sınırı içinde bilinçli olarak yapılmadı:

- **GPU / LLM inference** — proje CPU'da, küçük klasik ML modelleri (RandomForest gibi) için tasarlandı.
- **Kubernetes / orkestrasyon** — bu proje boyutu için Docker Compose yeterli.
- **Authentication / authorization** — geliştirme ortamı, localhost varsayımı; `/stats`, `/metrics` ve tahmin soketi kimlik doğrulaması olmadan açık.
- **Gerçek epoll / multi-process worker havuzu** — thread + kuyruk deseni öğrenme hedefi için yeterliydi; tam bir epoll implementasyonu ayrı bir mimari denemesi olurdu. Bunun somut sonucu: darboğaz hâlâ `NUM_SOCKET_WORKERS=4` — per-IP limiti bunu tek bir istemcinin tekeline almasını zorlaştırır ama havuzun kendisini büyütmez.
- **Birim testler** — zaman kısıtı nedeniyle Locust ile yük/entegrasyon testine ve CI'daki uçtan uca smoke test'e odaklanıldı, birim test yazılmadı.

## Gelecek geliştirmeler

- Model versiyonlama
- Aynı anda birden fazla modeli servis edebilme (parametrik model seçimi)
- Redis için persistence/auth (`requirepass`, AOF) — şu an açık ve kalıcı değil

## Öğrenilen şeyler

- TCP socket programlama (bare metal, framework olmadan)
- Producer-consumer deseni ve kuyruk-tabanlı iş dağıtımı
- Container orkestrasyon (Docker Compose, health check bağımlılıkları)
- ONNX ile framework-bağımsız model serving
- Backpressure ve yük yönetimi
- Bir sistemin gerçek darboğazının nerede olduğunu ölçerek bulmak — varsayarak değil
