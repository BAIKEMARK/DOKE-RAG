# -*- coding: utf-8 -*-

"""
最终综合分析脚本
================================

功能:
1. 读取“冠军赛”生成的详细排名CSV文件 (Championship_Rankings_Final.csv)。
2. 从三个维度对16个参数组合进行全局统计：
   - “夺冠”次数 (获得Rank 1的次数)
   - 平均排名
   - 平均胜场数
3. 生成一份最终的、排序好的综合分析报告CSV，并打印出核心结论。

如何使用:
1. 安装所需库: pip install pandas
2. 将此文件保存为 `最终分析脚本.py`。
3. 在下面的“配置区”指定 `CHAMPIONSHIP_CSV_PATH` 的正确路径。
4. 直接运行此Python脚本。
"""

import pandas as pd
from pathlib import Path
from doke_rag.config.paths import RESULTS_DIR, CSV_CHAMPIONSHIP

# --- 1. 配置区 (您需要在这里修改路径) ---

# 【修改这里】指向"冠军赛"生成的 Championship_Rankings_Final.csv 文件 - 从配置模块导入
CHAMPIONSHIP_CSV_PATH = CSV_CHAMPIONSHIP

# 【修改这里】最终综合报告的输出路径
OUTPUT_CSV_PATH = RESULTS_DIR / "Championship_Evaluation_Final" / "Final_Overall_Analysis.csv"

# --- 2. 主分析逻辑 ---


def analyze_final_results(filepath: str, output_path: Path):
    """
    读取冠军赛排名数据，并进行最终的综合统计分析。
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"错误: 未找到输入的排名文件: '{filepath}'")
        print("请先运行“冠军赛评估脚本.py”生成该文件。")
        return

    print(f"成功加载排名数据，共 {len(df)} 条记录。")
    print("-" * 30)

    # 创建每个参数组合的唯一标识符
    df["combination_id"] = (
        df["system"]
        + " (cs="
        + df["cosine"].astype(str)
        + ", tk="
        + df["top_k"].astype(str)
        + ")"
    )

    # --- 分析1: “夺冠”次数最多 ---
    df_rank_1 = df[df["rank"] == 1]
    first_place_counts = df_rank_1["combination_id"].value_counts().reset_index()
    first_place_counts.columns = ["combination_id", "first_place_wins"]

    # --- 分析2: 平均排名最高 ---
    average_ranks = df.groupby("combination_id")["rank"].mean().reset_index()
    average_ranks.columns = ["combination_id", "average_rank"]

    # --- 分析3: 平均胜场最多 ---
    average_win_counts = df.groupby("combination_id")["win_count"].mean().reset_index()
    average_win_counts.columns = ["combination_id", "average_win_count"]

    # --- 合并所有分析结果 ---
    # 从一个基础信息表开始，确保所有16个组合都被包含
    summary_df = (
        df[["combination_id", "system", "cosine", "top_k"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    summary_df = pd.merge(
        summary_df, first_place_counts, on="combination_id", how="left"
    )
    summary_df = pd.merge(summary_df, average_ranks, on="combination_id", how="left")
    summary_df = pd.merge(
        summary_df, average_win_counts, on="combination_id", how="left"
    )

    # 将NaN的夺冠次数填充为0
    summary_df["first_place_wins"] = (
        summary_df["first_place_wins"].fillna(0).astype(int)
    )

    # --- 排序以找出最终冠军 ---
    # 优先按“夺冠次数”降序排，其次按“平均排名”升序排，最后按“平均胜场”降序排
    final_summary = summary_df.sort_values(
        by=["first_place_wins", "average_rank", "average_win_count"],
        ascending=[False, True, False],
    ).reset_index(drop=True)

    # 保存到CSV
    final_summary.to_csv(output_path, index=False, encoding="utf-8-sig")

    # --- 打印核心结论 ---
    print("最终综合分析完成！")
    print(f"详细报告已保存至: {output_path.resolve()}")
    print("-" * 30)

    winner = final_summary.iloc[0]
    print("🏆 **综合总冠军** 🏆")
    print(f"根据多维度综合排序，表现最佳的组合是:")
    print(f"  - 系统 (System): {winner['system']}")
    print(f"  - 参数 (Params): cosine={winner['cosine']}, top_k={winner['top_k']}")
    print("\n其关键表现数据如下:")
    print(
        f"  - 🥇 **夺冠次数**: 在 {len(df['query'].unique())} 个问题中，获得了 {winner['first_place_wins']} 次第一名。"
    )
    print(f"  - 📊 **平均排名**: 所有问题中的平均排名为 {winner['average_rank']:.2f}。")
    print(
        f"  - 💪 **平均胜场**: 在每次循环赛中，平均赢得 {winner['average_win_count']:.2f} 场对决。"
    )
    print("-" * 30)

    print("\n完整排名报告预览 (Top 5):")
    print(final_summary.head(5).to_string(index=False))


# --- 脚本入口 ---
if __name__ == "__main__":
    analyze_final_results(CHAMPIONSHIP_CSV_PATH, OUTPUT_CSV_PATH)