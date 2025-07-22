import numpy as np
import os
import sys
from src.pointcloud_registration.utils import (
    load_point_cloud,
    find_nearest_neighbors_safe,
    compute_rmse,
)
from src.pointcloud_registration.pca_align import pca_align
from src.pointcloud_registration.icp_algorithm import icp
from src.pointcloud_registration.visualize import (
    plot_rmse_history,
    save_pointcloud_plot,
)


def compute_robust_rmse(
    aligned: np.ndarray, target: np.ndarray, top_ratio: float = 0.90
) -> float:
    indices = find_nearest_neighbors_safe(aligned, target)
    matched = target[indices]
    dists = np.sqrt(np.sum((aligned - matched) ** 2, axis=1))
    sorted_dists = np.sort(dists)
    k = int(len(dists) * top_ratio)
    return np.sqrt(np.mean(sorted_dists[:k] ** 2))


def log_progress(prefix: str, i: int, total: int) -> None:
    percent = (i + 1) / total * 100
    print(f"{prefix} [{i+1:6d}/{total}] {percent:6.2f}% 完了")


def main(N: int = 2000) -> None:
    os.makedirs("results_testpoints", exist_ok=True)
    print(f"=== Running ICP with N = {N} ===")

    source_full = load_point_cloud("data/会議室1_19_53_16.txt")
    target_full = load_point_cloud("data/会議室2_19_53_44 (1).txt")

    np.random.seed(42)
    source_sample = source_full[np.random.choice(len(source_full), N, replace=False)]
    target_sample = target_full[np.random.choice(len(target_full), N, replace=False)]

    source_sample_aligned, target_sample_aligned, R_pca, mean_pca = pca_align(
        source_sample, target_sample
    )

    aligned_sample, rmse_history, R_icp, t_icp, _ = icp(
        source_sample_aligned, target_sample_aligned, target_full, max_iterations=200
    )
    print(f"[Sample] Final RMSE: {rmse_history[-1]:.6f}")
    print(f"[Sample] Iterations: {len(rmse_history)}")

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

    plot_rmse_history(rmse_history, save_path=f"results_testpoints/rmse_plot_N{N}.png")
    save_pointcloud_plot(
        aligned_full, target_eval, save_path=f"results_testpoints/pointclouds_N{N}.png"
    )
    print("[LOG] 完了：results_testpoints/ に保存しました。")


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    main(N)
