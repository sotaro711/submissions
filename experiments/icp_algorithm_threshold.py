import numpy as np


def compute_rmse(src: np.ndarray, dst: np.ndarray) -> float:
    diff = src - dst
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))


def best_fit_transform(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    assert A.shape == B.shape
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = AA.T @ BB
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = centroid_B - R @ centroid_A
    return R, t


def icp_threshold(
    source: np.ndarray,
    target: np.ndarray,
    distance_threshold: float = 0.5,
    max_iterations: int = 50,
    tolerance: float = 1e-5,
) -> tuple[np.ndarray, list[float], np.ndarray, np.ndarray]:
    if source.shape[1] != 3 or target.shape[1] != 3:
        raise ValueError("Input point clouds must be (N, 3)")

    src = source.copy()
    rmse_history: list[float] = []
    R_total = np.eye(3)
    t_total = np.zeros(3)

    for i in range(max_iterations):
        dists = np.linalg.norm(src[:, np.newaxis, :] - target[np.newaxis, :, :], axis=2)
        indices = np.argmin(dists, axis=1)
        matched = target[indices]
        errors = np.linalg.norm(src - matched, axis=1)

        # thresholdで対応点を制限
        mask = errors < distance_threshold
        if np.sum(mask) < 10:
            print(f"[Threshold ICP] Too few inliers at iter {i}. Skipping update.")
            break

        filtered_src = src[mask]
        filtered_matched = matched[mask]

        R, t = best_fit_transform(filtered_src, filtered_matched)
        src = (R @ src.T).T + t
        R_total = R @ R_total
        t_total = R @ t_total + t

        rmse = compute_rmse(filtered_src, filtered_matched)
        rmse_history.append(rmse)

        print(f"[Threshold ICP] Iter {i}: RMSE = {rmse:.6f}")
        if i > 0 and abs(rmse_history[-2] - rmse) < tolerance:
            print(
                f"[Threshold ICP] Converged at iter {i} with delta RMSE {abs(rmse_history[-2] - rmse):.6e}"
            )
            break

    return src, rmse_history, R_total, t_total
