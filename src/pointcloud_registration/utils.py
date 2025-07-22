import numpy as np


def load_point_cloud(file_path: str) -> np.ndarray:
    """Loads a point cloud file and returns only the XYZ coordinates.

    Args:
        file_path (str): Path to the point cloud file.

    Returns:
        np.ndarray: N x 3 array of XYZ coordinates.
    """
    points = np.loadtxt(file_path)
    return points[:, :3]  # Use only x, y, z columns


def downsample_point_cloud(points: np.ndarray, num_samples: int) -> np.ndarray:
    """Randomly downsamples the point cloud to the given number of points.

    Args:
        points (np.ndarray): Original point cloud (N, 3)
        num_samples (int): Number of points to sample

    Returns:
        np.ndarray: Downsampled point cloud (num_samples, 3)
    """
    if len(points) <= num_samples:
        return points
    indices = np.random.choice(len(points), num_samples, replace=False)
    return points[indices]


def compute_rmse(source: np.ndarray, target: np.ndarray) -> float:
    """Computes RMSE between corresponding points.

    Args:
        source (np.ndarray): Source point cloud (N, 3)
        target (np.ndarray): Target point cloud (N, 3)

    Returns:
        float: Root mean squared error
    """
    return np.sqrt(np.mean(np.sum((source - target) ** 2, axis=1)))


def compute_robust_rmse(
    source: np.ndarray, target: np.ndarray, top_ratio: float = 0.9
) -> float:
    """Computes robust RMSE using only the top_ratio of closest point pairs.

    Args:
        source (np.ndarray): Source point cloud (N, 3)
        target (np.ndarray): Target point cloud (N, 3)
        top_ratio (float, optional): Ratio of points to keep. Defaults to 0.9.

    Returns:
        float: Robust root mean squared error
    """
    errors = np.sqrt(np.sum((source - target) ** 2, axis=1))
    sorted_errors = np.sort(errors)
    cutoff = int(len(errors) * top_ratio)
    trimmed_errors = sorted_errors[:cutoff]
    return np.sqrt(np.mean(trimmed_errors**2))


def find_nearest_neighbors_safe(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Finds the nearest neighbors in dst for each point in src, with progress logs.

    Args:
        src (np.ndarray): Source point cloud (N, 3)
        dst (np.ndarray): Target point cloud (M, 3)

    Returns:
        np.ndarray: Array of indices (length N), where each index corresponds to the nearest point in dst
    """
    indices = []
    total = len(src)
    for i, point in enumerate(src):
        if i % 1000 == 0 or i == total - 1:
            print(f"[Eval NN] {i+1}/{total} ({(i+1)/total*100:.2f}%) 完了")
        distances = np.sum((dst - point) ** 2, axis=1)
        indices.append(np.argmin(distances))
    return np.array(indices)


def estimate_rigid_transform(
    A: np.ndarray, B: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Estimates the optimal rigid transformation (rotation + translation) from A to B.

    Args:
        A (np.ndarray): Source points (N, 3)
        B (np.ndarray): Target points (N, 3)

    Returns:
        tuple[np.ndarray, np.ndarray]: Rotation matrix (3x3), translation vector (3,)
    """
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = centroid_B - R @ centroid_A
    return R, t
