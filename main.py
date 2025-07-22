import numpy as np
import os
from src.pointcloud_registration.utils import (
    load_point_cloud,
    find_nearest_neighbors_safe,
    compute_rmse,
)
from src.pointcloud_registration.pca_align_multi import pca_align_multi
from src.pointcloud_registration.icp_algorithm import icp
from src.pointcloud_registration.visualize import (
    plot_rmse_history,
    save_pointcloud_plot,
)


def compute_robust_rmse(
    aligned: np.ndarray, target: np.ndarray, top_ratio: float = 0.90
) -> float:
    """Compute the robust RMSE using top N% of smallest distances.

    Args:
        aligned (np.ndarray): Aligned source point cloud (Nx3).
        target (np.ndarray): Target point cloud (Nx3).
        top_ratio (float): Ratio of the smallest distances to include in RMSE calculation.

    Returns:
        float: Robust RMSE value.
    """
    indices = find_nearest_neighbors_safe(aligned, target)
    matched = target[indices]
    dists = np.sqrt(np.sum((aligned - matched) ** 2, axis=1))
    sorted_dists = np.sort(dists)
    k = int(len(dists) * top_ratio)
    return np.sqrt(np.mean(sorted_dists[:k] ** 2))


def log_progress(prefix: str, i: int, total: int) -> None:
    """Prints formatted progress log with percentage.

    Args:
        prefix (str): Message prefix.
        i (int): Current iteration number.
        total (int): Total number of iterations.
    """
    percent = (i + 1) / total * 100
    print(f"{prefix} [{i+1:6d}/{total}] {percent:6.2f}% 完了")


def main() -> None:
    """Main function to perform point cloud registration using multi-start PCA + ICP.

    Loads source and target point clouds, performs multi-start PCA alignment followed by ICP,
    selects the best result based on final RMSE, and evaluates full and robust RMSE on full data.

    Saves RMSE plot and aligned point cloud visualization.
    """
    os.makedirs("results_multistart", exist_ok=True)

    print("[LOG] 点群を読み込み中...")
    source_full = load_point_cloud("data/会議室1_19_53_16.txt")
    target_full = load_point_cloud("data/会議室2_19_53_44 (1).txt")

    N = 2000
    np.random.seed(42)
    source_sample = source_full[np.random.choice(len(source_full), N, replace=False)]
    target_sample = target_full[np.random.choice(len(target_full), N, replace=False)]

    print("[LOG] 複数PCA初期整列を実行中...")
    candidates = pca_align_multi(source_sample, target_sample)

    best_rmse = float("inf")
    best_result = None

    for i, (src_aligned, tgt_aligned, R_pca, mean_pca) in enumerate(candidates):
        aligned_sample, rmse_history, R_icp, t_icp, _ = icp(
            src_aligned, tgt_aligned, target_full, max_iterations=200
        )
        print(f"[Multi-PCA {i}] Final RMSE: {rmse_history[-1]:.6f}")
        if rmse_history[-1] < best_rmse:
            best_rmse = rmse_history[-1]
            best_result = (
                aligned_sample,
                rmse_history,
                R_icp,
                t_icp,
                R_pca,
                mean_pca,
                src_aligned,
                tgt_aligned,
            )

    (
        aligned_sample,
        rmse_history,
        R_icp,
        t_icp,
        R_pca,
        mean_pca,
        src_aligned,
        tgt_aligned,
    ) = best_result

    print(f"[Multi-PCA Best] Final RMSE: {rmse_history[-1]:.6f}")

    R_total = R_icp @ R_pca
    t_total = t_icp - R_icp @ R_pca @ mean_pca
    aligned_full = (R_total @ source_full.T).T + t_total

    print("[LOG] full target でRMSE評価中...")
    target_eval = target_full
    indices = []
    for i, point in enumerate(aligned_full):
        if i % 10000 == 0 or i == len(aligned_full) - 1:
            log_progress("[Eval NN]", i, len(aligned_full))
        dists = np.sum((target_eval - point) ** 2, axis=1)
        indices.append(np.argmin(dists))
    matched = target_eval[np.array(indices)]

    rmse_full = compute_rmse(aligned_full, matched)
    robust_rmse = compute_robust_rmse(aligned_full, target_eval, top_ratio=0.90)

    print(f"[Full] RMSE: {rmse_full:.6f}")
    print(f"[Robust] RMSE (top 90%): {robust_rmse:.6f}")

    print("[LOG] 結果を保存中...")
    plot_rmse_history(
        rmse_history, save_path="results_multistart/rmse_plot_multistart.png"
    )
    save_pointcloud_plot(
        aligned_full,
        target_eval,
        save_path="results_multistart/pointclouds_multistart.png",
    )
    print("[LOG] 完了：results_multistart/ に保存しました。")


if __name__ == "__main__":
    main()
