import numpy as np
import itertools


def pca_align_multi(
    source: np.ndarray, target: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """PCAにより複数の初期整列候補を生成する。

    Returns:
        list of tuples: 各候補に対し、(整列後ソース点群, 整列後ターゲット点群, 回転行列, ソース重心)
    """

    def get_pca_frame(pc: np.ndarray) -> np.ndarray:
        pc_centered = pc - np.mean(pc, axis=0)
        cov = np.cov(pc_centered.T)
        _, eigvecs = np.linalg.eigh(cov)
        return eigvecs[:, ::-1]  # 固定軸順（反転含む）

    mean_source = np.mean(source, axis=0)
    mean_target = np.mean(target, axis=0)
    R_source_base = get_pca_frame(source)
    R_target = get_pca_frame(target)

    results = []
    signs = list(itertools.product([-1, 1], repeat=3))

    for s in signs:
        R_source = R_source_base * s
        R = R_target @ R_source.T
        source_aligned = (R @ (source - mean_source).T).T
        target_aligned = target - mean_target
        results.append((source_aligned, target_aligned, R, mean_source))

    return results
