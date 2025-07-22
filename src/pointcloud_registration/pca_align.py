import numpy as np


def pca_align(
    source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """PCAにより初期整列を行う（反転対策済み）。

    Args:
        source (np.ndarray): ソース点群 (N, 3)
        target (np.ndarray): ターゲット点群 (N, 3)

    Returns:
        tuple: 整列後ソース点群, 整列後ターゲット点群, 回転行列, ソースの重心
    """

    def get_pca_frame(pc: np.ndarray) -> np.ndarray:
        pc_centered = pc - np.mean(pc, axis=0)
        cov = np.cov(pc_centered.T)
        _, eigvecs = np.linalg.eigh(cov)
        eigvecs = eigvecs[:, ::-1]  # 反転対策
        return eigvecs

    mean_source = np.mean(source, axis=0)
    mean_target = np.mean(target, axis=0)
    R_source = get_pca_frame(source)
    R_target = get_pca_frame(target)

    R = R_target @ R_source.T
    source_aligned = (R @ (source - mean_source).T).T
    target_aligned = target - mean_target

    return source_aligned, target_aligned, R, mean_source
