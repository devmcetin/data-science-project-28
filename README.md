# Data Science Project 28 — Müşteri Churn Baseline

**Modül**: ML-01 (Makine Öğrenmesine Giriş) • **Süre**: 2-3 saat

## 🎯 Proje Senaryosu

Bir telekomünikasyon şirketinde (TelcoTR) **junior data scientist** olarak işe başladın. CEO bir sabah toplantıda diyor:

> "Müşterilerimizin **%20'si** her ay aboneliğini iptal ediyor. Bu kabul edilemez. Bana hangi müşterilerin **iptal etme riski yüksek** söyleyen bir model lazım. Riskli müşterilere kampanya göndereceğiz."

İşte senin projen bu — bir **müşteri churn (kayıp) tahmin modeli** kuracaksın.

Bu projede ML-01 dersinde öğrendiğin **HER ŞEYİ** kullanacaksın:
- ✅ **Stratify** (dengesiz veri — %80 kal vs %20 ayrıl)
- ✅ **Pipeline** (data leakage'a karşı koruma)
- ✅ **Cross-validation** (tek split'e güvenme, 5-fold ile gerçek performans)
- ✅ **Overfit teşhisi** (train vs test acc gap)
- ✅ **K optimizasyonu** (en iyi KNN k değerini bulma)
- ✅ **predict_proba** (olasılığa göre tahmin — riskli müşterileri sırala)

## 📦 Proje Kurulumu

```bash
# Fork + clone
git clone <your-fork-url>
cd data-science-project-28

# Virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate          # Windows

# Dependencies
pip install -r requirements.txt

# Auto test runner (dosya değişince çalışır)
python watch.py

# Manuel test
pytest tests/test_question.py -v
```

## 🔑 Kaizu Bağlantısı — `kaizu_config.py`

Skorunun Kaizu hesabına yazılması için **`kaizu_config.py`** dosyasını aç ve **`USER_ID`** alanını kendi user_id'nle değiştir:

```python
USER_ID = 0      # ← Kaizu profilinden alıp buraya yaz
PROJECT_ID = 708 # ← Bu projeye ait, dokunma
```

User_id'ni Kaizu profilinden bulabilirsin (Profile → Settings → User ID).

Skor göndermek için tüm testleri toplu çalıştırmalısın:

```bash
python tests/test_question.py
```

Bu komut tüm testleri çalıştırır, **passed/total oranını otomatik Kaizu'ya gönderir**. Geliştirme sırasında `pytest -v` kullanmaya devam edebilirsin (skor göndermez).

## 📋 Görevler (`tasks/task_manager.py`)

`task_manager.py` dosyasındaki fonksiyonları sırayla doldur. Her task altta testler pass olana kadar düzenlenmeli.

1. **`load_churn_data()`** — Sentetik telco churn verisini yükle. 2000 müşteri, %80 Stay + %20 Churn. 5 özellik: tenure_months, monthly_charges, total_charges, num_services, support_calls

2. **`explore_class_balance(y, target_names)`** — Sınıf dağılımını dict olarak döndür. **Dengesiz veriyi göstermek kritik!**

3. **`split_data_stratified(X, y)`** — `train_test_split` ile böl. **stratify=y ŞART** — dengesiz veride sınıf oranı korunmalı. test_size=0.2, random_state=42.

4. **`build_pipeline(k=5)`** — `Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(k))])` döndür. Pipeline data leakage'a karşı otomatik koruma sağlar.

5. **`train_pipeline(pipe, X_train, y_train)`** — Pipeline'ı eğit, eğitilmiş pipeline'ı döndür.

6. **`evaluate_with_cv(pipe, X, y, cv=5)`** — `cross_val_score` ile 5-fold CV. Dict döndür: `{'mean': ..., 'std': ..., 'scores': [...]}`. Std küçükse model kararlı demektir.

7. **`detect_overfit(train_acc, test_acc, threshold=0.10)`** — Gap > threshold ise overfit alarmı. Dict döndür: `{'overfit': True/False, 'gap': ...}`. Gap %10'dan büyükse alarm ver.

8. **`find_best_k(X_train, y_train, k_values)`** — Her k için 5-fold CV ortalama skoru hesapla, en iyi k'yı döndür. Dict: `{'best_k': ..., 'best_score': ..., 'all_scores': {k: score, ...}}`.

9. **`predict_with_proba(pipe, X_new, threshold=0.5)`** — `predict_proba` kullan, olasılık eşiği `threshold`'tan büyükse 'Churn', değilse 'Stay'. **Niye eşik?** Dengesiz veride %50 default eşik kötü olabilir, daha düşük eşikle riskli müşterileri yakalamak gerek. Dict döndür: `{'predictions': array, 'probabilities': array}`.

10. **`run_full_pipeline()`** — Yukarıdaki fonksiyonları birleştirip uçtan uca bir analiz yap. Sonuç dict olarak döndür: load → split → train → CV → best_k → predict.

## 🧪 Testler

Test dosyası: `tests/test_question.py` (16 test)

Tümü pass olmalı:
- Veri yükleme + dengesiz sınıf doğrulama
- Stratify çalışıyor mu (test setinde de %80/%20)
- Pipeline doğru kurulmuş mu
- CV ortalama 0.7+ (dengesiz veride accuracy yanıltıcı, F1 daha iyi ama bu temel proje)
- Overfit detection mantığı
- find_best_k düzgün çalışıyor
- predict_proba çıktıları doğru formatta

## 📊 Beklenen Sonuçlar

```
Sınıf dağılımı: {'Stay': 1600, 'Churn': 400}  (%80 / %20 dengesiz)
Train: 1600, Test: 400 (stratified — Test'te de 320 Stay + 80 Churn)
5-fold CV: ~0.85 ± 0.02 (dengeli model)
En iyi k: 5-15 arası (sweet spot)
Overfit: gap < %5 (sağlıklı)
```

## 💡 İpuçları

- **stratify'ı UNUTMA** — yoksa test setine 0 churn düşebilir
- **Pipeline kullan** — `scaler.fit_transform(X)` sonra split = data leakage (fake skor)
- **CV'siz tek split şüpheli** — random_state şanslı/şanssız fark yaratabilir
- **predict_proba > predict** — özellikle dengesiz verilerde olasılığı sıralayıp riskli müşterileri seç

## 🎓 Öğrenme Çıktıları

Bu projeyi bitirdiğinde:
- Dengesiz veri ile başa çıkmayı (stratify, predict_proba threshold) bilirsin
- Pipeline'ın data leakage'ı nasıl önlediğini anlarsın
- CV ortalama + std okumayı bilirsin (kararlı model = düşük std)
- Hyperparameter (k) optimizasyonunu CV ile doğru yapabilirsin
- Bir baseline ML modeli kurup performansını profesyonelce raporlayabilirsin

## 🚫 Dikkat

- `tests/test_question.py` dosyasını **değiştirme**
- `random_state=42` değerini değiştirme (testler fail olur)
- `_solution/` klasörü yok (DB'de saklanır, dersin haftası geçince açılır)
- Dokunabileceğin **2 dosya**: `tasks/task_manager.py` (kodu yaz) + `kaizu_config.py` (sadece USER_ID)
