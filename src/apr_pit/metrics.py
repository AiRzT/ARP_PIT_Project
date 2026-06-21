from __future__ import annotations

import numpy as np


def field_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    epsilon: float = 1.0e-12,
) -> dict[str, float]:
    """Return manuscript-style pointwise and global field errors.

    Arrays must already describe the same physical field on the same grid. NaN
    or infinite reference cells are excluded, which is useful for masked FDS
    slice exports.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if prediction.shape != reference.shape:
        raise ValueError(
            f"Prediction/reference shape mismatch: {prediction.shape} != {reference.shape}"
        )
    valid = np.isfinite(prediction) & np.isfinite(reference)
    if not np.any(valid):
        raise ValueError("No finite prediction/reference pairs are available")

    predicted = prediction[valid]
    target = reference[valid]
    difference = predicted - target
    mae = np.mean(np.abs(difference))
    rmse = np.sqrt(np.mean(difference**2))
    relative_l2 = np.linalg.norm(difference) / max(np.linalg.norm(target), epsilon)
    centered_target = target - np.mean(target)
    r2 = 1.0 - np.sum(difference**2) / max(np.sum(centered_target**2), epsilon)
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "max_absolute_error": float(np.max(np.abs(difference))),
        "relative_l2": float(relative_l2),
        "r2": float(r2),
        "valid_points": int(valid.sum()),
    }
