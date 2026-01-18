
# -*- coding: utf-8 -*-

"""
最终排名报告可视化脚本
================================

功能:
1. 无需命令行，直接在脚本顶部的“配置区”指定文件路径。
2. 读取“冠军赛”后生成的最终分析报告 (Final_Overall_Analysis.csv)。
3. 生成一张包含三个子图的大图，每个子图都是一个 4x4 矩阵热力图，分别代表三种最终排名标准：
   - “夺冠次数” (First Place Wins)
   - “平均排名” (Average Rank)
   - “平均胜场” (Average Win Count)
4. 每个单元格的颜色代表该项指标的强弱，中央则标注出达成该成绩的系统名称。
5. 整个图表采用顶级期刊的科研美术风格。

如何使用:
1. 安装所需库: pip install pandas matplotlib seaborn
2. 在下面的“配置区”修改好输入文件和输出文件夹的路径。
3. 直接运行此Python脚本。
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from doke_rag.config.paths import RESULTS_DIR

# --- 1. 配置区 (您需要在这里修改路径) ---

# 【修改这里】指向你之前生成的 Final_Overall_Analysis.csv 文件 - 使用相对路径
ANALYSIS_CSV_PATH = RESULTS_DIR / "Championship_Evaluation_Final" / "Final_Overall_Analysis.csv"

# 【修改这里】图表的输出文件夹 - 使用相对路径
OUTPUT_DIRECTORY = RESULTS_DIR / "experiment_evaluation" / "Figures"

# --- 2. 核心绘图函数 ---


def create_ranking_matrix_heatmap(
    ax,
    df: pd.DataFrame,
    value_col: str,
    title: str,
    lower_is_better: bool = False,
):
    """
    创建一个4x4的矩阵热力图，颜色代表指标强弱，标注代表系统名称。
    """
    ax.spines[["right", "top"]].set_visible(False)

    # 准备用于绘图的两个核心矩阵：一个用于颜色，一个用于标注
    try:
        # 用于颜色的数值矩阵
        value_pivot = df.pivot(
            index="top_k", columns="cosine", values=value_col
        ).sort_index(ascending=False)
        # 用于中央标注的系统名称矩阵
        annot_pivot = df.pivot(
            index="top_k", columns="cosine", values="system"
        ).sort_index(ascending=False)
    except Exception as e:
        ax.set_title(title, fontsize=16, weight="bold", pad=12)
        ax.text(0.5, 0.5, f"Pivot failed:\n{e}", ha="center", va="center")
        return

    # 根据指标特性选择色板（值越大越好，还是越小越好）
    cmap = "viridis_r" if lower_is_better else "viridis"

    sns.heatmap(
        value_pivot,
        ax=ax,
        annot=annot_pivot,  # 关键：使用系统名称矩阵进行标注
        fmt="",  # 标注内容已经是字符串，无需格式化
        linewidths=1.0,
        linecolor="white",
        cmap=cmap,
        cbar=True,
        cbar_kws={"label": f"Metric Value ({value_col})"},
        annot_kws={"size": 10, "weight": "bold"},
    )

    ax.set_title(title, fontsize=16, weight="bold", pad=12)
    ax.set_xlabel("Cosine Threshold", fontsize=14)
    ax.set_ylabel("Top K", fontsize=14)
    ax.tick_params(axis="y", rotation=0)


# --- 3. 主执行逻辑 ---
def main():
    """主执行函数"""
    mpl.rcParams["font.family"] = "DejaVu Sans"

    input_file = Path(ANALYSIS_CSV_PATH)
    if not input_file.exists():
        print(f"错误: 找不到输入的分析文件: '{input_file}'")
        print("请先运行“最终分析脚本.py”生成该文件。")
        return

    df_final = pd.read_csv(input_file)
    output_dir = Path(OUTPUT_DIRECTORY)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("成功加载最终排名数据，正在生成三合一总结矩阵图...")

    # --- 创建一个 1x3 的大图，包含三个子图 ---
    fig, axes = plt.subplots(1, 3, figsize=(28, 8), constrained_layout=True)
    fig.suptitle(
        "Final Ranking Analysis Across Different Metrics", fontsize=22, weight="bold"
    )

    # 子图1: 夺冠次数 (越高越好)
    create_ranking_matrix_heatmap(
        axes[0],
        df_final,
        value_col="first_place_wins",
        title="First Place Wins",
    )

    # 子图2: 平均排名 (越低越好)
    create_ranking_matrix_heatmap(
        axes[1],
        df_final,
        value_col="average_rank",
        title="Average Rank",
        lower_is_better=True,  # 告知函数这个指标是越小越好
    )

    # 子图3: 平均胜场 (越高越好)
    create_ranking_matrix_heatmap(
        axes[2],
        df_final,
        value_col="average_win_count",
        title="Average Win Count",
    )

    # 保存最终的图像
    filename = output_dir / "Final_Ranking_Matrix_Overview_rename.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"\n已保存最终总结矩阵图: {filename}")
    plt.show()


if __name__ == "__main__":
    main()
