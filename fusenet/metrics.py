from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, recall_score


def _to_numpy(x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    return arr


def pearson_corr(pred, true) -> float:
    pred = _to_numpy(pred)
    true = _to_numpy(true)
    if pred.size < 2 or np.std(pred) < 1e-12 or np.std(true) < 1e-12:
        return 0.0
    return float(np.corrcoef(pred, true)[0, 1])


def mean_absolute_error(pred, true) -> float:
    pred = _to_numpy(pred)
    true = _to_numpy(true)
    return float(np.mean(np.abs(pred - true)))


def _binarize_by_sign(values: np.ndarray, zero_positive: bool) -> np.ndarray:
    return values >= 0 if zero_positive else values > 0


def binary_metrics(
    pred,
    true,
    *,
    zero_positive: bool,
    exclude_zero_truth: bool = False,
) -> Dict[str, float]:
    pred = _to_numpy(pred)
    true = _to_numpy(true)
    if exclude_zero_truth:
        keep = true != 0
        pred = pred[keep]
        true = true[keep]
    if true.size == 0:
        return {"acc2": 0.0, "f1": 0.0}
    y_true = _binarize_by_sign(true, zero_positive=zero_positive)
    y_pred = _binarize_by_sign(pred, zero_positive=zero_positive)
    return {
        "acc2": float(accuracy_score(y_true, y_pred) * 100.0),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100.0),
    }


def paper_binary_metrics(pred, true) -> Dict[str, float]:
    """Official MOSI/MOSEI binary protocols.

    Has0 keeps samples whose ground-truth score is 0 and counts zero as the
    non-negative class. No0 removes zero-label samples and uses the positive
    vs. negative split.
    """
    pred_arr = _to_numpy(pred)
    true_arr = _to_numpy(true)
    has0 = binary_metrics(pred_arr, true_arr, zero_positive=True, exclude_zero_truth=False)
    no0 = binary_metrics(pred_arr, true_arr, zero_positive=False, exclude_zero_truth=True)
    return {
        "acc2_has0": has0["acc2"],
        "f1_has0": has0["f1"],
        "acc2_no0": no0["acc2"],
        "f1_no0": no0["f1"],
        "has0_support": int(true_arr.size),
        "no0_support": int((true_arr != 0).sum()),
        "zero_support": int((true_arr == 0).sum()),
        # Default aliases follow the standard No0 MOSI/MOSEI Acc-2/F1 protocol.
        "acc2": no0["acc2"],
        "f1": no0["f1"],
        "acc2_nz": no0["acc2"],
        "f1_nz": no0["f1"],
    }


def quantized_accuracy(pred, true, classes: int, label_min: float, label_max: float) -> float:
    pred = _to_numpy(pred)
    true = _to_numpy(true)
    pred = np.clip(pred, label_min, label_max)
    true = np.clip(true, label_min, label_max)
    scale = (classes - 1) / (label_max - label_min)
    pred_q = np.rint((pred - label_min) * scale).astype(np.int64)
    true_q = np.rint((true - label_min) * scale).astype(np.int64)
    return float(accuracy_score(true_q, pred_q) * 100.0)


def threshold_accuracy(pred, true, thresholds: Sequence[float], *, clip_min: float, clip_max: float, right: bool) -> float:
    pred = np.clip(_to_numpy(pred), clip_min, clip_max)
    true = np.clip(_to_numpy(true), clip_min, clip_max)
    cuts = np.asarray(thresholds, dtype=np.float64)
    pred_q = np.digitize(pred, cuts, right=right)
    true_q = np.digitize(true, cuts, right=right)
    return float(accuracy_score(true_q, pred_q) * 100.0)


def regression_report(pred, true, dataset: str, label_min: float, label_max: float) -> Dict[str, float]:
    report = {
        "mae": mean_absolute_error(pred, true),
        "corr": pearson_corr(pred, true),
    }
    dataset = dataset.lower()
    if dataset in {"mosi", "mosei"}:
        report.update(paper_binary_metrics(pred, true))
        report["acc7"] = quantized_accuracy(pred, true, 7, -3.0, 3.0)
    elif dataset == "simsv2":
        report.update(binary_metrics(pred, true, zero_positive=False, exclude_zero_truth=False))
        report.update({f"{k}_nz": v for k, v in binary_metrics(pred, true, zero_positive=False, exclude_zero_truth=True).items()})
        report["acc3"] = threshold_accuracy(pred, true, [-0.1, 0.1], clip_min=-1.0, clip_max=1.0, right=True)
        report["acc5"] = threshold_accuracy(pred, true, [-0.7, -0.1, 0.1, 0.7], clip_min=-1.0, clip_max=1.0, right=True)
    else:
        report.update(binary_metrics(pred, true, zero_positive=True, exclude_zero_truth=False))
        report.update({f"{k}_nz": v for k, v in binary_metrics(pred, true, zero_positive=False, exclude_zero_truth=True).items()})
        report["acc3"] = threshold_accuracy(pred, true, [-0.1, 0.1], clip_min=-1.0, clip_max=1.0, right=True)
        report["acc5"] = threshold_accuracy(pred, true, [-0.7, -0.1, 0.1, 0.7], clip_min=-1.0, clip_max=1.0, right=True)
    return report


def classification_report(pred, true, labels: Sequence[str]) -> Dict[str, object]:
    pred_arr = np.asarray(pred, dtype=np.int64).reshape(-1)
    true_arr = np.asarray(true, dtype=np.int64).reshape(-1)
    label_ids = np.arange(len(labels), dtype=np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_arr,
        pred_arr,
        labels=label_ids,
        zero_division=0,
    )
    return {
        "overall": {
            "test_samples": int(true_arr.size),
            "WA_accuracy": float(accuracy_score(true_arr, pred_arr)),
            "UA_macro_recall": float(recall_score(true_arr, pred_arr, labels=label_ids, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(true_arr, pred_arr, labels=label_ids, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(true_arr, pred_arr, labels=label_ids, average="weighted", zero_division=0)),
        },
        "per_class": [
            {
                "label": str(label),
                "support": int(support[idx]),
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
            }
            for idx, label in enumerate(labels)
        ],
    }
