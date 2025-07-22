import numpy as np
from src.pointcloud_registration.utils import (
    find_nearest_neighbors_safe,
    estimate_rigid_transform,
    compute_rmse,
)


def icp(
    source: np.ndarray,
    target: np.ndarray,
    target_eval: np.ndarray,
    max_iterations: int = 50,
    tolerance: float = 1e-5,
) -> tuple[np.ndarray, list[float], np.ndarray, np.ndarray, list[float]]:
    src = source.copy()
    dst = target.copy()
    R_total = np.eye(3)
    t_total = np.zeros(3)
    prev_error = float("inf")
    rmse_history = []
    full_rmse_history = []

    for i in range(max_iterations):
        indices = find_nearest_neighbors_safe(src, dst)
        matched_dst = dst[indices]
        R, t = estimate_rigid_transform(src, matched_dst)
        src = (R @ src.T).T + t
        R_total = R @ R_total
        t_total = R @ t_total + t

        rmse = np.sqrt(np.mean(np.sum((src - matched_dst) ** 2, axis=1)))
        rmse_history.append(rmse)

        # Full RMSE (against evaluation target point cloud)
        eval_indices = find_nearest_neighbors_safe(src, target_eval)
        matched_eval = target_eval[eval_indices]
        full_rmse = compute_rmse(src, matched_eval)
        full_rmse_history.append(full_rmse)

        if abs(prev_error - rmse) < tolerance:
            break
        prev_error = rmse

    return src, rmse_history, R_total, t_total, full_rmse_history
