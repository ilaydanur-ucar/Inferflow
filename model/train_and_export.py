# model/train_and_export.py
#
# Bu script iki iş yapıyor:
#   1) Basit bir ML modeli eğitmek (RandomForest, Iris veri setiyle)
#   2) O modeli ONNX formatına çevirip diske kaydetmek
#
# ONNX'e çevirmemizin sebebi: sonraki adımda bu modeli onnxruntime ile
# CPU üzerinde hızlıca çalıştıracağız — model_runner.py içindeki predict()
# fonksiyonu bu .onnx dosyasını okuyacak.

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
# RandomForestClassifier: birden fazla karar ağacının oy birliğiyle
# sınıflandırma yapan bir model. Seçme sebebimiz: eğitimi saniyeler
# sürüyor (GPU gerektirmiyor) VE ONNX'e çevirimi skl2onnx tarafından
# sağlam/güvenilir şekilde destekleniyor (ör. Isolation Forest gibi
# bazı modellerin ONNX dönüşümü hâlâ hatalı/sorunlu; RandomForest
# bu sorunu yaşamıyor).

from skl2onnx import to_onnx
# skl2onnx: scikit-learn modellerini ONNX formatına çeviren kütüphane.
# to_onnx(): "bana eğitilmiş modeli ver, ONNX karşılığını üreteyim" diyen
# ana fonksiyon.

from skl2onnx.common.data_types import FloatTensorType
# FloatTensorType: ONNX'e "modele girecek verinin şekli/tipi bu olacak"
# demek için kullanılan bir tanım nesnesi. ONNX statik bir format
# olduğu için, modelin girdisinin kaç sütunlu (kaç özellik) ve hangi
# veri tipinde (float) olacağını ona AÇIKÇA söylememiz gerekiyor —
# scikit-learn'de böyle bir zorunluluk yoktu, ONNX'e özgü bir adım bu.

X, y = load_iris(return_X_y=True)

clf = RandomForestClassifier(n_estimators=50, random_state=42) 

clf.fit(X, y)
initial_type = [("float_input", FloatTensorType([None, X.shape[1]]))]
# Bu satır ONNX'e modelin girdisinin NASIL bir şey olacağını anlatıyor.
# FloatTensorType([None, X.shape[1]]):
#   - X.shape[1] = 4 (Iris'in 4 özelliği) → "her girdi satırı 4 sayı içerecek"
#   - None = "kaç satır (kaç örnek) geleceğini şimdiden bilmiyorum, esnek olsun"
#     (yani tek bir tahmin de gönderebilirsin, 100 tanesini birden de —
#     buna "batch boyutu" denir, None onu sabitlemiyor.)

onx = to_onnx(clf, initial_types=initial_type)

with open("model/model.onnx", "wb") as f:
    # "wb" = write binary → ONNX dosyaları metin değil, ikili (binary)
    # formatta olduğu için "b" (binary) modunda açıyoruz.
    f.write(onx.SerializeToString())
    # SerializeToString(): ONNX modelini (protobuf formatında) ham
    # byte dizisine çeviriyor, biz de onu dosyaya yazıyoruz.
    # Sonuç: model/model.onnx adında, diskte duran bir dosya.

print("Model eğitildi ve model/model.onnx olarak kaydedildi.")
print(f"Örnek girdi şekli: {X.shape}, sınıf sayısı: {len(set(y))}")
# Bu iki print, script bittiğinde ne olduğunu terminalde görebilmen için.
# X.shape → (150, 4): 150 örnek, 4 özellik.
# len(set(y)) → 3: kaç farklı sınıf (çiçek türü) olduğu.