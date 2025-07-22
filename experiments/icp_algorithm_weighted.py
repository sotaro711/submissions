import numpy as np


def compute_weighted_rmse(
    src: np.ndarray, dst: np.ndarray, weights: np.ndarray
) -> float:
    """Compute weighted RMSE between two point clouds."""
    diff = src - dst
    weighted_sum = np.sum(weights * np.sum(diff**2, axis=1))
    return np.sqrt(weighted_sum / np.sum(weights))


def compute_weighted_best_fit_transform(
    A: np.ndarray, B: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the best-fit transform with per-point weights.

    Args:
        A (np.ndarray): Source point cloud (N, 3)
        B (np.ndarray): Target point cloud (N, 3)
        weights (np.ndarray): Weights for each point (N,)

    Returns:
        tuple[np.ndarray, np.ndarray]: Rotation matrix, translation vector
    """
    assert A.shape == B.shape
    assert A.shape[0] == weights.shape[0]

    # Normalize weights
    weights = weights / np.sum(weights)

    centroid_A = np.average(A, axis=0, weights=weights)
    centroid_B = np.average(B, axis=0, weights=weights)

    AA = A - centroid_A
    BB = B - centroid_B

    H = (AA * weights[:, np.newaxis]).T @ BB
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A
    return R, t


def icp_weighted(
    source: np.ndarray,
    target: np.ndarray,
    max_iterations: int = 50,
    tolerance: float = 1e-5,
) -> tuple[np.ndarray, list[float], np.ndarray, np.ndarray]:
    """Perform the weighted Iterative Closest Point algorithm.

    Args:
        source (np.ndarray): Source point cloud (N, 3)
        target (np.ndarray): Target point cloud (N, 3)
        max_iterations (int): Maximum iterations
        tolerance (float): Convergence threshold

    Returns:
        tuple: aligned source, RMSE history, total R, total t
    """
    if source.shape[1] != 3 or target.shape[1] != 3:
        raise ValueError("Point clouds must be of shape (N, 3)")

    src = source.copy()
    rmse_history: list[float] = []
    R_total = np.eye(3)
    t_total = np.zeros(3)

    for i in range(max_iterations):
        dists = np.linalg.norm(src[:, np.newaxis, :] - target[np.newaxis, :, :], axis=2)
        indices = np.argmin(dists, axis=1)
        matched = target[indices]

        distances = np.linalg.norm(src - matched, axis=1)
        weights = 1.0 / (distances + 1e-8)  # avoid divide by zero

        R, t = compute_weighted_best_fit_transform(src, matched, weights)
        src = (R @ src.T).T + t
        R_total = R @ R_total
        t_total = R @ t_total + t

        rmse = compute_weighted_rmse(src, matched, weights)
        rmse_history.append(rmse)

        if i > 0 and abs(rmse_history[-2] - rmse) < tolerance:
            print(
                f"[Weighted ICP] Converged at iteration {i} with delta RMSE {abs(rmse_history[-2] - rmse):.6e}"
            )
            break

        print(f"[Weighted ICP] Iter {i}: RMSE = {rmse:.6f}")

    return src, rmse_history, R_total, t_total
