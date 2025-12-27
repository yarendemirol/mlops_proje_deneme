from sklearn.metrics import accuracy_score


def run_quality_check(y_true, y_pred, threshold=0.85):
    """
    Döküman III.3: Continued Model Evaluation & Algorithmic Fallback [cite: 45, 49]
    """
    acc = accuracy_score(y_true, y_pred)
    print(f"\n[CME] Current Model Accuracy: {acc:.4f}")

    if acc < threshold:
        print("!!! WARNING: Performance Degradation Detected !!!")
        print("[FALLBACK] Action: Triggering Algorithmic Fallback strategy.")[cite: 49]
        return "TRIGGER_FALLBACK"

    print("[SUCCESS] Model performance is healthy.")
    return "HEALTHY"