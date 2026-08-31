FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Container dışından (host'tan) bağlanabilmek için sunucu tüm
# arayüzleri dinlemeli; 127.0.0.1 sadece container'ın kendi içinden
# erişilebilir olurdu, -p ile yayınlanan porta host'tan ulaşılamazdı.
ENV HOST=0.0.0.0
# TTY olmadığında Python stdout'u buffer'lar; sunucu hiç çıkmadığı
# için "listening" logu asla flush edilmezdi (docker logs'ta görünmez).
ENV PYTHONUNBUFFERED=1

EXPOSE 9000 8080

# Container ayağa kalkarken önce modeli üret (model/model.onnx repoda
# yok, .gitignore'da), sonra sunucuyu başlat.
CMD ["sh", "-c", "python model/train_and_export.py && python server.py"]
