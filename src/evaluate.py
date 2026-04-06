"""
ScamShield Evaluation Suite — Comprehensive model evaluation with 7+ metrics.

Metrics reported:
  - F1 (macro & binary)
  - ROC-AUC
  - MCC (Matthews Correlation Coefficient)
  - Brier Score
  - Recall (sensitivity)
  - Precision
  - Confusion Matrix
  - Classification Report
  - McNemar's test (model comparison)
"""

import numpy as np
from sklearn.metrics import (
    f1_score, roc_auc_score, matthews_corrcoef,
    brier_score_loss, precision_score, recall_score,
    confusion_matrix, classification_report
)


def full_evaluation(model, X_test, y_test, model_name="ScamShield"):
    """
    Run comprehensive evaluation and print all metrics.

    Returns dict of all computed metrics.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "f1_macro":   f1_score(y_test, y_pred, average="macro"),
        "f1_binary":  f1_score(y_test, y_pred),
        "roc_auc":    roc_auc_score(y_test, y_proba),
        "mcc":        matthews_corrcoef(y_test, y_pred),
        "brier":      brier_score_loss(y_test, y_proba),
        "recall":     recall_score(y_test, y_pred),
        "precision":  precision_score(y_test, y_pred),
    }

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Safe", "Scam"])

    print(f"\n{'=' * 55}")
    print(f"  Evaluation: {model_name}")
    print(f"{'=' * 55}")
    print(f"  F1 (macro):    {metrics['f1_macro']:.4f}")
    print(f"  F1 (binary):   {metrics['f1_binary']:.4f}")
    print(f"  ROC-AUC:       {metrics['roc_auc']:.4f}")
    print(f"  MCC:           {metrics['mcc']:.4f}")
    print(f"  Brier Score:   {metrics['brier']:.4f}")
    print(f"  Recall:        {metrics['recall']:.4f}")
    print(f"  Precision:     {metrics['precision']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"                Pred Safe  Pred Scam")
    print(f"    Actual Safe   {cm[0][0]:>6}     {cm[0][1]:>6}")
    print(f"    Actual Scam   {cm[1][0]:>6}     {cm[1][1]:>6}")
    print(f"\n{report}")

    return metrics


def per_language_evaluation(model, X_test, y_test, languages, model_name="ScamShield"):
    """Evaluate model performance per language."""
    print(f"\n{'=' * 55}")
    print(f"  Per-Language Evaluation: {model_name}")
    print(f"{'=' * 55}")
    print(f"  {'Language':<10} {'F1':>8} {'Recall':>8} {'Precision':>8} {'N':>6}")
    print(f"  {'-' * 42}")

    results = {}
    for lang in sorted(set(languages)):
        mask = np.array(languages) == lang
        if mask.sum() < 10:
            continue

        y_pred = model.predict(X_test[mask])
        f1 = f1_score(y_test[mask], y_pred, zero_division=0)
        rec = recall_score(y_test[mask], y_pred, zero_division=0)
        prec = precision_score(y_test[mask], y_pred, zero_division=0)

        results[lang] = {"f1": f1, "recall": rec, "precision": prec, "n": int(mask.sum())}
        print(f"  {lang:<10} {f1:>8.4f} {rec:>8.4f} {prec:>8.4f} {mask.sum():>6}")

    return results


def mcnemar_test(y_true, y_pred_a, y_pred_b, model_a_name, model_b_name):
    """
    McNemar's test for statistical significance between two models.

    Tests whether the disagreements between models are significantly different.
    """
    try:
        from statsmodels.stats.contingency_tables import mcnemar as mcnemar_stat

        # Build contingency table
        b = int(((y_pred_a == y_true) & (y_pred_b != y_true)).sum())
        c = int(((y_pred_a != y_true) & (y_pred_b == y_true)).sum())

        table = [[0, b], [c, 0]]
        result = mcnemar_stat(table, exact=True)

        print(f"\n  McNemar's Test: {model_a_name} vs {model_b_name}")
        print(f"    b={b} (A correct, B wrong)")
        print(f"    c={c} (A wrong, B correct)")
        print(f"    p-value: {result.pvalue:.6f}")
        print(f"    Significant (p<0.05): {result.pvalue < 0.05}")
        return {"b": b, "c": c, "pvalue": float(result.pvalue),
                "significant": bool(result.pvalue < 0.05)}

    except ImportError:
        print("  [WARNING] statsmodels not installed. Skipping McNemar's test.")
        return None


def error_analysis(model, X_test, y_test, texts, n_examples=10):
    """
    Analyze misclassified samples — shows what the model gets wrong.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # False negatives (missed scams — most dangerous)
    fn_mask = (y_test == 1) & (y_pred == 0)
    fn_indices = np.where(fn_mask)[0]

    print(f"\n{'=' * 55}")
    print(f"  Error Analysis")
    print(f"{'=' * 55}")
    print(f"\n  False Negatives (Missed Scams): {fn_mask.sum()}")
    for i in fn_indices[:n_examples]:
        print(f"    [{y_proba[i]:.3f}] {texts[i][:100]}...")

    # False positives (false alarms)
    fp_mask = (y_test == 0) & (y_pred == 1)
    fp_indices = np.where(fp_mask)[0]

    print(f"\n  False Positives (False Alarms): {fp_mask.sum()}")
    for i in fp_indices[:n_examples]:
        print(f"    [{y_proba[i]:.3f}] {texts[i][:100]}...")

    return {
        "false_negatives": int(fn_mask.sum()),
        "false_positives": int(fp_mask.sum()),
        "total_errors": int(fn_mask.sum() + fp_mask.sum()),
        "error_rate": float((fn_mask.sum() + fp_mask.sum()) / len(y_test)),
    }
