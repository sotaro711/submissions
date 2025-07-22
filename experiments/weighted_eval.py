import numpy as np
from src.pointcloud_registration.utils import (
    load_point_cloud,
    find_nearest_neighbors_safe,
    compute_rmse,
    compute_robust_rmse,
)
from src.pointcloud_registration.pca_align import pca_align
from src.pointcloud_registration.icp_algorithm_weighted import icp_weighted
from src.pointcloud_registration.visualize import (
    plot_rmse_history,
    save_pointcloud_plot,
)


def log_progress(prefix: str, i: int, total: int) -> None:
    """進捗ログを出力する"""
    percent = (i + 1) / total * 100
    print(f"{prefix} [{i+1:6d}/{total}] {percent:6.2f}% 完了")


def main():
    print("[LOG] 点群を読み込み中...")
    source_full = load_point_cloud("data/会議室1_19_53_16.txt")
    target_full = load_point_cloud("data/会議室2_19_53_44 (1).txt")
    print(f"[LOG] source_full 点数: {len(source_full)}")
    print(f"[LOG] target_full 点数: {len(target_full)}")

    # ダウンサンプリング
    N = 2200
    np.random.seed(42)
    source_sample = source_full[np.random.choice(len(source_full), N, replace=False)]
    target_sample = target_full[np.random.choice(len(target_full), N, replace=False)]

    # PCA初期整列
    print("[LOG] PCA初期整列...")
    source_sample_aligned, target_sample_aligned, R_pca, mean_pca = pca_align(
        source_sample, target_sample
    )

    # ICP（weighted）
    print("[LOG] Weighted ICP を実行...")
    aligned_sample, rmse_history, R_icp, t_icp = icp_weighted(
        source_sample_aligned, target_sample_aligned, max_iterations=200
    )
    print(f"[Sample] Final RMSE: {rmse_history[-1]:.6f}")
    print(f"[Sample] Iterations: {len(rmse_history)}")

    # フル点群への変換適用
    print("[LOG] フル点群に変換を適用中...")
    R_total = R_icp @ R_pca
    t_total = t_icp - R_icp @ R_pca @ mean_pca
    aligned_full = (R_total @ source_full.T).T + t_total

    # 評価（full target）
    print("[LOG] full target を用いて RMSE 評価中...")
    target_eval = target_full
    indices = []
    for i, point in enumerate(aligned_full):
        if i % 10000 == 0 or i == len(aligned_full) - 1:
            log_progress("[Eval NN]", i, len(aligned_full))
        dists = np.sum((target_eval - point) ** 2, axis=1)
        indices.append(np.argmin(dists))
    matched = target_eval[np.array(indices)]

    rmse_full = compute_rmse(aligned_full, matched)
    robust_rmse = compute_robust_rmse(aligned_full, matched, top_ratio=0.90)

    print(f"[Full] RMSE on full target: {rmse_full:.6f}")
    print(f"[Robust] RMSE (top 90%):    {robust_rmse:.6f}")

    # 結果保存
    print("[LOG] 結果画像を保存中...")
    plot_rmse_history(rmse_history, save_path="results/rmse_plot_weighted.png")
    save_pointcloud_plot(
        aligned_full, target_eval, save_path="results/pointclouds_weighted.png"
    )
    print("[LOG] 完了: 全ての結果を保存しました。")


if __name__ == "__main__":
    main()
