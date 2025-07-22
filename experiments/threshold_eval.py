import numpy as np
import os
from src.pointcloud_registration.utils import (
    load_point_cloud,
    find_nearest_neighbors_safe,
    compute_rmse,
)
from src.pointcloud_registration.pca_align import pca_align
from src.pointcloud_registration.icp_algorithm_threshold import icp_threshold
from src.pointcloud_registration.visualize import (
    plot_rmse_history,
    save_pointcloud_plot,
)


def compute_robust_rmse(
    source: np.ndarray, target: np.ndarray, top_ratio: float = 0.90
) -> float:
    indices = find_nearest_neighbors_safe(source, target)
    matched = target[indices]
    errors = np.sqrt(np.sum((source - matched) ** 2, axis=1))
    sorted_errors = np.sort(errors)
    cutoff = int(len(errors) * top_ratio)
    return np.sqrt(np.mean(sorted_errors[:cutoff] ** 2))


def log_progress(prefix: str, i: int, total: int) -> None:
    percent = (i + 1) / total * 100
    print(f"{prefix} [{i+1:6d}/{total}] {percent:6.2f}% 完了")


def main() -> None:
    os.makedirs("results_threshold", exist_ok=True)

    print("[LOG] 点群を読み込み中...")
    source_full = load_point_cloud("data/会議室1_19_53_16.txt")
    target_full = load_point_cloud("data/会議室2_19_53_44 (1).txt")

    print(f"[LOG] source_full 点数: {len(source_full)}")
    print(f"[LOG] target_full 点数: {len(target_full)}")

    N = 2000
    np.random.seed(42)
    source_sample = source_full[np.random.choice(len(source_full), N, replace=False)]
    target_sample = target_full[np.random.choice(len(target_full), N, replace=False)]

    print("[LOG] PCAによる初期整列中...")
    source_sample_aligned, target_sample_aligned, R_pca, mean_pca = pca_align(
        source_sample, target_sample
    )

    print("[LOG] Threshold ICP 実行中...")
    aligned_sample, rmse_history, R_icp, t_icp = icp_threshold(
        source_sample_aligned,
        target_sample_aligned,
        distance_threshold=0.5,
        max_iterations=200,
    )
    print(f"[Threshold Sample] Final RMSE: {rmse_history[-1]:.6f}")
    print(f"[Threshold Sample] Iterations: {len(rmse_history)}")

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
        rmse_history, save_path="results_threshold/rmse_plot_threshold.png"
    )
    save_pointcloud_plot(
        aligned_full,
        target_eval,
        save_path="results_threshold/pointclouds_threshold.png",
    )
    print("[LOG] 完了：results_threshold/ に保存されました。")


if __name__ == "__main__":
    main()
