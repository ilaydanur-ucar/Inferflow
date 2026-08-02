# model_runner.py
#
# Bu dosyanın tek görevi: model/model.onnx dosyasını yükleyip,
# geri kalan kodun (server.py) kolayca çağırabileceği basit bir
# predict() fonksiyonu sunmak.
#
# Neden ayrı bir dosya? Çünkü "modeli nasıl çalıştırıyorum" detayını
# (onnxruntime, session, girdi/çıktı isimleri) sunucu kodundan izole
# ediyoruz. server.py sadece predict(girdi) çağıracak, arkada ne
# döndüğünü bilmesine gerek yok. Bu, "ayrı sorumluluk" (separation
# of concerns) denen bir mühendislik prensibi.

import numpy as np
import onnxruntime as ort
# onnxruntime: ONNX formatındaki modelleri ÇALIŞTIRAN motor (kütüphane).
# skl2onnx modeli ONNX formatına ÇEVİRMİŞTİ (dönüştürücü);
# onnxruntime ise o dosyayı okuyup gerçek tahmin işlemini YAPAN taraf.
# İkisi farklı iş yapan iki farklı araç.

# Bunu fonksiyonun İÇİNE değil, dosyanın en üstüne koyuyoruz çünkü
# model yükleme (diskten okuma + hazırlama) nispeten yavaş bir iş.
# Eğer predict() her çağrıldığında modeli yeniden yüklersek, her
# istek çok daha yavaş olurdu. Burada bir kez yüklüyoruz, sonra
# gelen her istek aynı yüklü modeli (aynı "session"ı) kullanıyor.
# İleride (Hafta 2) worker süreçleri de bu dosyayı import ettiğinde
# her worker kendi sürecinde bu satırı bir kez çalıştıracak.

_session = ort.InferenceSession(
    "model/model.onnx",
    providers=["CPUExecutionProvider"],
)
# InferenceSession: modeli belleğe yükleyip "artık tahmin yapmaya
# hazır" hâle getiren nesne. Bir kere kurulur, sonra tekrar tekrar
# kullanılır (tıpkı bir veritabanı bağlantısı gibi düşünebilirsin).
#
# providers=["CPUExecutionProvider"]: onnxruntime'a "bu modeli CPU
# üzerinde çalıştır" diyoruz (alternatifi GPU olurdu, bizde yok zaten
# — bu proje bilinçli olarak CPU'ya göre tasarlandı).

_input_name = _session.get_inputs()[0].name
_output_name = _session.get_outputs()[0].name
# ONNX modeline veri verirken ve sonucu alırken, modelin girdi/çıktı
# "isimlerini" bilmemiz gerekiyor (train_and_export.py'de girdiye
# "float_input" adını vermiştik, hatırlarsan).
# Bunu elle yazmak yerine session'dan otomatik SORUYORUZ — böylece
# isim değişse bile (örn. modeli değiştirdiğinde) bu kod kendini
# otomatik ayarlıyor, elle güncellemene gerek kalmıyor.
# Alt çizgiyle (_session, _input_name) başlayan isimler Python'da
# "bu bir iç/özel değişken, dışarıdan doğrudan kullanma" anlamına
# gelen bir kural (zorunlu değil, bir anlaşma/convention).


def predict(features):
    """
    features: düz bir sayı listesi, örn. [5.1, 3.5, 1.4, 0.2]
              (Iris modelimiz için 4 sayı bekliyor.)

    Dönüş: modelin tahmini (sınıf ve/veya olasılıklar; ONNX modelinin
    ürettiği ham çıktı listeye çevrilmiş hâliyle).
    """

    x = np.asarray(features, dtype=np.float32).reshape(1, -1)
    # 1) np.asarray(..., dtype=np.float32):
    #    Gelen Python listesini numpy dizisine çeviriyoruz VE tipini
    #    float32 yapıyoruz. Neden float32? Çünkü ONNX modelini
    #    FloatTensorType ile (float32 anlamına gelir) tanımlamıştık —
    #    tipler uyuşmazsa onnxruntime hata verir.
    #
    # 2) .reshape(1, -1):
    #    Modelimiz "(satır_sayısı, 4)" şeklinde bir tablo bekliyor
    #    (train_and_export.py'deki FloatTensorType([None, 4])
    #    hatırlarsan). Bize gelen [5.1, 3.5, 1.4, 0.2] düz bir liste
    #    (tek boyutlu), modelin istediği şekil ise iki boyutlu.
    #    reshape(1, -1) diyor ki: "bunu 1 satırlık bir tabloya çevir,
    #    sütun sayısını sen kendin hesapla (-1 = otomatik)".
    #    Sonuç: [[5.1, 3.5, 1.4, 0.2]] şeklinde, (1, 4) boyutunda.

    out = _session.run([_output_name], {_input_name: x})
    # session.run(...) asıl tahmini burada yapıyor.
    # İlk argüman ([_output_name]): "hangi çıktıları istiyorum" listesi
    #   (bir modelin birden fazla çıktısı olabilir, biz sadece birini
    #   istiyoruz, o yüzden tek elemanlı liste).
    # İkinci argüman ({_input_name: x}): "girdi ismi: girdi verisi"
    #   şeklinde bir sözlük. ONNX, girdileri isimle eşleştirerek alır
    #   (pozisyona göre değil), bu yüzden _input_name'i kullanıyoruz.
    # Dönüş (out): bir liste — her istenen çıktı için bir eleman.
    #   Biz tek çıktı istediğimiz için out[0] bizim asıl sonucumuz.

    return out[0].tolist()
    # out[0]: numpy array formatında sonuç (örn. array([1])).
    # .tolist(): numpy array'i normal Python listesine çeviriyoruz.
    #   Neden? Çünkü ileride bu sonucu JSON'a çevirip ağdan
    #   göndereceğiz (server.py'de) — JSON, numpy tiplerini DEĞİL,
    #   sade Python tiplerini (list, int, float, str) anlar.
    #   Bu satırı atlarsak, json.dumps() hata verirdi.




if __name__ == "__main__":
    ornek = [5.1, 3.5, 1.4, 0.2]  # gerçek bir Iris çiçeğinin ölçüleri
    sonuc = predict(ornek)
    print(f"Girdi: {ornek}")
    print(f"Tahmin: {sonuc}")