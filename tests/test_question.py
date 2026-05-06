import pytest
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tasks.task_manager import (
    load_churn_data,
    explore_class_balance,
    split_data_stratified,
    build_pipeline,
    train_pipeline,
    evaluate_with_cv,
    detect_overfit,
    find_best_k,
    predict_with_proba,
    run_full_pipeline,
)


# 1. load_churn_data
def test_load_churn_data_returns_dict():
    data = load_churn_data()
    assert isinstance(data, dict)
    assert set(data.keys()) == {"X", "y", "feature_names", "target_names"}


def test_load_churn_data_shapes_and_balance():
    data = load_churn_data()
    assert data["X"].shape == (2000, 5)
    assert data["y"].shape == (2000,)
    assert data["target_names"] == ["Stay", "Churn"]
    # Sınıf dağılımı yaklaşık %80 / %20
    n_stay = int((data["y"] == 0).sum())
    n_churn = int((data["y"] == 1).sum())
    assert 1500 <= n_stay <= 1700  # ~1600 ± tolerans
    assert 300 <= n_churn <= 500   # ~400 ± tolerans


# 2. explore_class_balance
def test_explore_class_balance_format():
    data = load_churn_data()
    balance = explore_class_balance(data["y"], data["target_names"])
    assert isinstance(balance, dict)
    assert set(balance.keys()) == {"Stay", "Churn"}
    assert isinstance(balance["Stay"], int)
    assert balance["Stay"] + balance["Churn"] == 2000


# 3. split_data_stratified
def test_split_data_sizes():
    data = load_churn_data()
    X_train, X_test, y_train, y_test = split_data_stratified(data["X"], data["y"])
    assert X_train.shape[0] == 1600
    assert X_test.shape[0] == 400


def test_split_data_stratified_ratio():
    """stratify=y → train ve test'te %80/%20 oranı korunmalı"""
    data = load_churn_data()
    X_train, X_test, y_train, y_test = split_data_stratified(data["X"], data["y"])
    # Test setinde de %80/%20 olmalı
    test_churn_ratio = (y_test == 1).sum() / len(y_test)
    assert 0.18 <= test_churn_ratio <= 0.22  # %20 ± tolerans


# 4. build_pipeline
def test_build_pipeline_structure():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier

    pipe = build_pipeline(k=5)
    assert isinstance(pipe, Pipeline)
    # İlk adım StandardScaler, ikinci KNN olmalı
    assert isinstance(pipe.steps[0][1], StandardScaler)
    assert isinstance(pipe.steps[1][1], KNeighborsClassifier)
    assert pipe.steps[1][1].n_neighbors == 5


# 5. train_pipeline
def test_train_pipeline_fits():
    data = load_churn_data()
    X_train, X_test, y_train, y_test = split_data_stratified(data["X"], data["y"])
    pipe = build_pipeline(k=5)
    fitted = train_pipeline(pipe, X_train, y_train)
    # Fit edildikten sonra predict yapabilmeli
    preds = fitted.predict(X_test[:5])
    assert len(preds) == 5


# 6. evaluate_with_cv
def test_evaluate_with_cv_returns_dict():
    data = load_churn_data()
    pipe = build_pipeline(k=5)
    result = evaluate_with_cv(pipe, data["X"], data["y"], cv=5)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"mean", "std", "scores"}
    assert len(result["scores"]) == 5
    assert 0.7 <= result["mean"] <= 1.0  # makul performans
    assert 0.0 <= result["std"] <= 0.1   # kararlı olmalı


# 7. detect_overfit
def test_detect_overfit_true():
    result = detect_overfit(train_acc=0.98, test_acc=0.60, threshold=0.10)
    assert result["overfit"] is True
    assert abs(result["gap"] - 0.38) < 0.01


def test_detect_overfit_false():
    result = detect_overfit(train_acc=0.85, test_acc=0.84, threshold=0.10)
    assert result["overfit"] is False
    assert abs(result["gap"] - 0.01) < 0.01


# 8. find_best_k
def test_find_best_k_format():
    data = load_churn_data()
    X_train, _, y_train, _ = split_data_stratified(data["X"], data["y"])
    result = find_best_k(X_train, y_train, k_values=[3, 5, 10])
    assert isinstance(result, dict)
    assert set(result.keys()) == {"best_k", "best_score", "all_scores"}
    assert result["best_k"] in [3, 5, 10]
    assert set(result["all_scores"].keys()) == {3, 5, 10}


def test_find_best_k_picks_best():
    """En yüksek skora sahip k seçilmeli"""
    data = load_churn_data()
    X_train, _, y_train, _ = split_data_stratified(data["X"], data["y"])
    result = find_best_k(X_train, y_train, k_values=[1, 5, 50])
    # best_k all_scores'ta en yüksek değere sahip olan olmalı
    assert result["all_scores"][result["best_k"]] == result["best_score"]
    assert result["best_score"] == max(result["all_scores"].values())


# 9. predict_with_proba
def test_predict_with_proba_format():
    data = load_churn_data()
    X_train, X_test, y_train, _ = split_data_stratified(data["X"], data["y"])
    pipe = build_pipeline(k=5)
    pipe = train_pipeline(pipe, X_train, y_train)

    result = predict_with_proba(pipe, X_test[:10], threshold=0.5)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"predictions", "probabilities"}
    assert len(result["predictions"]) == 10
    assert len(result["probabilities"]) == 10
    # Predictions sadece 'Stay' veya 'Churn' olmalı
    for p in result["predictions"]:
        assert p in ["Stay", "Churn"]
    # Probabilities 0-1 arasında olmalı
    for p in result["probabilities"]:
        assert 0.0 <= p <= 1.0


def test_predict_with_proba_threshold_logic():
    """threshold=0.0 → hepsi Churn, threshold=1.0 → hepsi Stay"""
    data = load_churn_data()
    X_train, X_test, y_train, _ = split_data_stratified(data["X"], data["y"])
    pipe = build_pipeline(k=5)
    pipe = train_pipeline(pipe, X_train, y_train)

    low = predict_with_proba(pipe, X_test[:20], threshold=0.0)
    high = predict_with_proba(pipe, X_test[:20], threshold=1.0)

    # threshold=0 → her olasılık > 0 → hepsi Churn (proba > 0 olduğunda)
    # En azından düşük threshold daha çok Churn üretmeli
    n_churn_low = sum(1 for p in low["predictions"] if p == "Churn")
    n_churn_high = sum(1 for p in high["predictions"] if p == "Churn")
    assert n_churn_low >= n_churn_high


# 10. run_full_pipeline
def test_run_full_pipeline_keys():
    result = run_full_pipeline()
    assert isinstance(result, dict)
    expected = {
        "class_balance",
        "best_k",
        "cv_mean",
        "cv_std",
        "train_acc",
        "test_acc",
        "overfit_check",
        "sample_predictions",
        "sample_probabilities",
    }
    assert expected.issubset(result.keys())


def test_run_full_pipeline_values():
    result = run_full_pipeline()
    assert 0.7 <= result["cv_mean"] <= 1.0
    assert 0.0 <= result["cv_std"] <= 0.1
    assert isinstance(result["best_k"], int)
    assert len(result["sample_predictions"]) == 5
    assert len(result["sample_probabilities"]) == 5


# ──────────────────────────────────────────────────────
# Kaizu skor gönderimi — bu kısma DOKUNMA
# ──────────────────────────────────────────────────────

import requests


def _send_score(user_score):
    """Kaizu API'sine skor gönder. user_id ve project_id kaizu_config'ten gelir."""
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    try:
        from kaizu_config import USER_ID, PROJECT_ID
    except ImportError:
        print("⚠️  kaizu_config.py bulunamadı — skor gönderilmeyecek.")
        return

    if USER_ID == 0:
        print("⚠️  kaizu_config.py'de USER_ID=0 — kendi ID'ni yazmadın, skor gönderilmeyecek.")
        return

    url = "https://kaizu-api-8cd10af40cb3.herokuapp.com/projectLog"
    payload = {
        "user_id": USER_ID,
        "project_id": PROJECT_ID,
        "user_score": user_score,
        "is_auto": True,
    }
    try:
        r = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
        if r.status_code in (200, 201):
            print(f"✅ Skor gönderildi: {user_score}")
        else:
            print(f"⚠️  Skor gönderilemedi (HTTP {r.status_code})")
    except Exception as e:
        print(f"⚠️  Skor gönderilirken hata: {e}")


class _ResultCollector:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            if report.passed:
                self.passed += 1
            elif report.failed:
                self.failed += 1


def run_tests():
    """Tüm testleri çalıştır + skoru Kaizu'ya gönder."""
    collector = _ResultCollector()
    pytest.main([os.path.dirname(__file__), "-q"], plugins=[collector])
    total = collector.passed + collector.failed
    if total == 0:
        print("Hiç test çalışmadı.")
        return
    user_score = round((collector.passed / total) * 100, 2)
    print(f"\n📊 Toplam başarılı : {collector.passed}/{total}")
    print(f"📊 Skor            : {user_score}")
    _send_score(user_score)


if __name__ == "__main__":
    run_tests()
