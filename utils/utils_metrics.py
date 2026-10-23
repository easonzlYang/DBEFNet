"""Binary segmentation metrics accumulated over valid pixels."""

import numpy as np


METRIC_NAMES = (
    "OA", "mIoU", "Macro_Precision", "Macro_Recall", "Macro_F1", "Kappa",
    "IoU_background", "IoU_UV", "Precision_background", "Precision_UV",
    "Recall_background", "Recall_UV", "F1_background", "F1_UV",
)


class SegmentationMetrics:
    """Accumulate one global confusion matrix and derive segmentation metrics.

    Rows are reference labels and columns are predictions. Precision, recall,
    F1 and IoU are first computed for background and urban village separately.
    Their arithmetic means form the macro metrics.
    Label value 255 is excluded from the confusion matrix.
    """

    class_names = ("background", "UV")

    def __init__(self):
        self.confusion_matrix = np.zeros((2, 2), dtype=np.int64)

    def update(self, prediction, label):
        prediction, label = np.asarray(prediction), np.asarray(label)
        if prediction.shape != label.shape:
            raise ValueError("Prediction and label dimensions must match")
        valid = label != 255
        label, prediction = label[valid], prediction[valid]
        if not np.isin(label, [0, 1]).all() or not np.isin(prediction, [0, 1]).all():
            raise ValueError("Metrics require binary class indices")
        index = label.astype(np.int64) * 2 + prediction.astype(np.int64)
        self.confusion_matrix += np.bincount(index.ravel(), minlength=4).reshape(2, 2)

    def merge(self, other):
        """Merge another independently accumulated confusion matrix."""
        self.confusion_matrix += other.confusion_matrix

    @staticmethod
    def _safe_divide(numerator, denominator):
        return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float),
                         where=denominator != 0)

    def compute(self):
        matrix = self.confusion_matrix.astype(np.float64)
        total = matrix.sum()
        if total == 0:
            raise ValueError("No valid labeled pixels to evaluate")
        true_positive = np.diag(matrix)
        false_positive = matrix.sum(axis=0) - true_positive
        false_negative = matrix.sum(axis=1) - true_positive
        precision = self._safe_divide(true_positive, true_positive + false_positive)
        recall = self._safe_divide(true_positive, true_positive + false_negative)
        iou = self._safe_divide(true_positive, true_positive + false_positive + false_negative)
        f1 = self._safe_divide(2 * precision * recall, precision + recall)
        oa = float(true_positive.sum() / total)
        expected = float((matrix.sum(axis=0) * matrix.sum(axis=1)).sum() / total ** 2)
        kappa = float((oa - expected) / (1 - expected)) if expected != 1 else 0.0
        return {
            "OA": oa, "mIoU": float(iou.mean()),
            "Macro_Precision": float(precision.mean()),
            "Macro_Recall": float(recall.mean()), "Macro_F1": float(f1.mean()),
            "Kappa": kappa,
            "IoU_background": float(iou[0]), "IoU_UV": float(iou[1]),
            "Precision_background": float(precision[0]), "Precision_UV": float(precision[1]),
            "Recall_background": float(recall[0]), "Recall_UV": float(recall[1]),
            "F1_background": float(f1[0]), "F1_UV": float(f1[1]),
        }
