import matplotlib.pyplot as plt
import numpy as np


def plot_point_clouds(
    source: np.ndarray, target: np.ndarray, title: str = "Point Clouds"
) -> None:
    """表示用：2次元点群をプロットして画面に表示"""
    plt.figure()
    plt.scatter(target[:, 0], target[:, 1], c="red", s=1, label="Target")
    plt.scatter(source[:, 0], source[:, 1], c="blue", s=1, label="Source")
    plt.legend()
    plt.axis("equal")
    plt.title(title)
    plt.show()


def save_pointcloud_plot(
    source: np.ndarray, target: np.ndarray, save_path: str, title: str = ""
) -> None:
    """保存用：2次元点群を画像として保存"""
    plt.figure()
    plt.scatter(target[:, 0], target[:, 1], c="red", s=1, label="Target")
    plt.scatter(source[:, 0], source[:, 1], c="blue", s=1, label="Source")
    plt.legend()
    plt.axis("equal")
    if title:
        plt.title(title)
    plt.savefig(save_path)
    plt.close()


def plot_rmse_history(rmse_history: list[float], save_path: str = None) -> None:
    """RMSEの収束グラフを描画し、必要なら保存

    Args:
        rmse_history (list[float]): 各イテレーションでのRMSE
        save_path (str, optional): 保存先のパス。指定しなければ 'rmse_plot_multistart.png' に保存
    """
    plt.figure()
    plt.plot(rmse_history, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("RMSE")
    plt.title("ICP Convergence")
    plt.grid(True)

    # Y軸の範囲を動的に調整
    if rmse_history:
        min_rmse = min(rmse_history)
        max_rmse = max(rmse_history)
        margin = (max_rmse - min_rmse) * 0.1 if max_rmse > min_rmse else 0.01
        plt.ylim(min_rmse - margin, max_rmse + margin)

    plt.xlim(0, len(rmse_history))

    # 保存処理
    if save_path is None:
        save_path = "rmse_plot_multistart.png"  # main.py と同じ場所に保存
    plt.savefig(save_path)
    plt.close()


def save_pointcloud_plot_3d(
    source: np.ndarray, target: np.ndarray, filename: str
) -> None:
    """3次元点群を可視化して画像として保存する

    Args:
        source (np.ndarray): ソース点群 (N, 3)
        target (np.ndarray): ターゲット点群 (M, 3)
        filename (str): 保存ファイルパス
    """
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(target[:, 0], target[:, 1], target[:, 2], s=0.1, c="r", label="Target")
    ax.scatter(source[:, 0], source[:, 1], source[:, 2], s=0.1, c="b", label="Source")
    ax.legend()
    ax.set_title("3D Pointcloud Alignment")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.savefig(filename)
    plt.close()
