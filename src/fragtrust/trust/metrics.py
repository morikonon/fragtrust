from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def classification_summary(
    y: np.ndarray, pred: np.ndarray, class_names: list[str]
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray]:
    labels = np.arange(len(class_names))
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    summary = {
        "n": int(len(y)),
        "accuracy": float((y == pred).mean()),
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall.mean()),
    }
    per_class = pd.DataFrame(
        {"class": class_names, "precision": precision, "recall": recall, "f1": f1, "support": support}
    )
    return summary, per_class, confusion_matrix(y, pred, labels=labels)
