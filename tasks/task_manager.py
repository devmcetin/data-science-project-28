"""
DS-28 — Müşteri Churn Baseline
Bir telekom şirketinde junior data scientist olarak müşteri churn (kayıp)
tahmin modeli kuruyorsun. ML-01'de öğrendiğin tüm araçları kullanacaksın.

Her fonksiyonun pass kısmını doldur. Testleri çalıştır, hepsi geçene kadar
iterate et: `python watch.py` veya `pytest tests/test_question.py -v`
"""


# 1. Sentetik churn verisini yükle
def load_churn_data():
    """
    sklearn.datasets.make_classification kullanarak sentetik telco verisi üret.

    Parametreler:
        n_samples=2000, n_features=5, n_informative=4, n_redundant=1
        n_classes=2, weights=[0.8, 0.2]   ← %80 Stay, %20 Churn (dengesiz!)
        random_state=42

    Returns:
        dict: {
            'X': feature matrix (numpy array, shape (2000, 5)),
            'y': target vector (numpy array, shape (2000,) — 0=Stay, 1=Churn),
            'feature_names': ['tenure_months', 'monthly_charges',
                              'total_charges', 'num_services', 'support_calls'],
            'target_names': ['Stay', 'Churn']
        }
    """
    pass


# 2. Sınıf dengesini incele
def explore_class_balance(y, target_names):
    """
    Her sınıftan kaç tane olduğunu hesapla.

    Args:
        y: target vector (numpy array)
        target_names: ['Stay', 'Churn']

    Returns:
        dict: {'Stay': 1600, 'Churn': 400}  (sayılar int olmalı)

    Not: Dengesiz veri var (%80 / %20). Bu fark stratify gerektirir!
    """
    pass


# 3. Stratified train/test split
def split_data_stratified(X, y):
    """
    train_test_split kullan:
    - test_size=0.2
    - random_state=42
    - stratify=y   ← ŞART! Dengesiz veride sınıf oranını koru.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    pass


# 4. Pipeline kur (StandardScaler + KNN)
def build_pipeline(k=5):
    """
    sklearn.pipeline.Pipeline kullan:
    - 1. adım: StandardScaler
    - 2. adım: KNeighborsClassifier(n_neighbors=k)

    Pipeline neden? Veri leakage'a karşı otomatik koruma — scaler train'de
    fit, test'te transform yapar (her CV fold'unda da doğru çalışır).

    Returns:
        sklearn.pipeline.Pipeline
    """
    pass


# 5. Pipeline'ı eğit
def train_pipeline(pipe, X_train, y_train):
    """
    pipe.fit(X_train, y_train) çağır, eğitilmiş pipeline'ı dön.

    Returns:
        sklearn.pipeline.Pipeline (fit edilmiş)
    """
    pass


# 6. Cross-validation ile değerlendir
def evaluate_with_cv(pipe, X, y, cv=5):
    """
    cross_val_score ile k-fold CV yap.

    Args:
        pipe: pipeline (henüz fit edilmemiş — cross_val_score kendisi fit eder)
        X, y: tüm veri (CV kendi içinde böler)
        cv: fold sayısı (default 5)

    Returns:
        dict: {
            'mean': float (ortalama accuracy),
            'std':  float (standart sapma — kararlı model = düşük std),
            'scores': list (5 fold skoru)
        }
    """
    pass


# 7. Overfit teşhisi
def detect_overfit(train_acc, test_acc, threshold=0.10):
    """
    Train ile test accuracy arasındaki gap'e bak.
    Eğer gap > threshold ise overfit alarmı.

    Args:
        train_acc: float (0-1)
        test_acc: float (0-1)
        threshold: float (default 0.10 = %10)

    Returns:
        dict: {'overfit': bool, 'gap': float}
    """
    pass


# 8. En iyi k'yı bul (CV ile)
def find_best_k(X_train, y_train, k_values):
    """
    Her k değeri için 5-fold CV ortalama skorunu hesapla, en iyi k'yı bul.

    Args:
        X_train, y_train: eğitim verisi
        k_values: list of int, örn [1, 3, 5, 10, 20, 50]

    Returns:
        dict: {
            'best_k': int (en iyi k),
            'best_score': float (en iyi CV ortalama),
            'all_scores': {k: score, ...} dict
        }

    İpucu: Her k için build_pipeline(k=k) ile yeni pipe kurabilirsin.
    """
    pass


# 9. predict_proba ile threshold'lu tahmin
def predict_with_proba(pipe, X_new, threshold=0.5):
    """
    Sadece predict() değil, predict_proba() kullan.
    Churn olasılığı (sınıf 1) > threshold ise 'Churn', değilse 'Stay'.

    Niye threshold? Dengesiz veride %50 default eşik kötü olabilir.
    Daha düşük threshold (örn 0.3) ile daha çok riskli müşteri yakalanır.

    Args:
        pipe: eğitilmiş pipeline
        X_new: yeni müşterilerin özellikleri (shape (n, 5))
        threshold: float (default 0.5)

    Returns:
        dict: {
            'predictions': numpy array of strings ('Stay' veya 'Churn'),
            'probabilities': numpy array of float (her müşterinin churn olasılığı)
        }

    İpucu: pipe.predict_proba(X_new)[:, 1] → sınıf 1 (Churn) olasılığı.
           numpy.where ile threshold'a göre 'Churn'/'Stay' ata.
    """
    pass


# 10. Tüm pipeline'ı uçtan uca çalıştır
def run_full_pipeline():
    """
    Yukarıdaki fonksiyonları sırayla kullanarak uçtan uca analiz yap:
    1. Veriyi yükle
    2. Sınıf dengesi çıkar
    3. Stratified split
    4. find_best_k ile k optimizasyonu (k_values=[1,3,5,10,20,50])
    5. En iyi k ile pipeline kur, eğit
    6. CV ile değerlendir
    7. Train vs test → overfit kontrolü
    8. İlk 5 test örneği için predict_with_proba

    Returns:
        dict: {
            'class_balance': dict,
            'best_k': int,
            'cv_mean': float,
            'cv_std': float,
            'train_acc': float,
            'test_acc': float,
            'overfit_check': dict,
            'sample_predictions': list (5 element),
            'sample_probabilities': list (5 element)
        }
    """
    pass


if __name__ == "__main__":
    result = run_full_pipeline()
    print("📊 Pipeline Sonuçları:")
    for k, v in result.items():
        print(f"  {k}: {v}")
