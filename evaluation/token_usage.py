# -*- coding: utf-8 -*-
import json
from pathlib import Path
import pandas as pd
from typing import Dict, Any, Tuple
from doke_rag.config.paths import RESULTS_DIR

# --- 1. 配置区 (请根据你的实际路径修改) ---

# 定义三个系统的文件路径和 Token 键名
# 注意：以下路径使用相对路径，相对于项目根目录
SYSTEM_CONFIGS = [
    {
        "label": "DOKE-RAG",
        "path": RESULTS_DIR / "batch_experiment/cs0.2_tk40/stru_mech_result.json",
        "input_key": "input_tokens",
        "output_key": "output_tokens",
        "nested": False,  # 标记是否需要处理嵌套结构
    },
    {
        "label": "LightRAG",
        "path": RESULTS_DIR / "batch_experiment/cs0.2_tk80/lightrag_result.json",
        "input_key": "input_tokens",
        "output_key": "output_tokens",
        "nested": False,
    },
    {
        "label": "GraphRAG",
        "path": RESULTS_DIR / "GraphRAG_local_search_result.json",
        # GraphRAG 的 Token 位于 'token_usage' 字典内
        "input_key": "prompt_tokens",
        "output_key": "completion_tokens",
        "nested": True,
    },
]

# --- 2. 核心函数 ---


def get_tokens(item: Dict[str, Any], config: Dict[str, Any]) -> Tuple[int, int]:
    """
    根据配置从单个问答记录中提取输入和输出 Token 数量。
    处理嵌套和非嵌套结构。
    """
    input_t, output_t = 0, 0

    try:
        if config["nested"]:
            # 处理嵌套结构，例如 GraphRAG
            token_data = item.get("token_usage", {})
            input_t = token_data.get(config["input_key"], 0)
            output_t = token_data.get(config["output_key"], 0)
        else:
            # 处理非嵌套结构，例如 DOKE-RAG, LightRAG
            input_t = item.get(config["input_key"], 0)
            output_t = item.get(config["output_key"], 0)
    except Exception:
        # 如果任何一步解析失败，返回 0
        pass

        # 确保返回整数
    return int(input_t), int(output_t)


def analyze_tokens(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    加载文件，计算总 Token 数、总问答数和平均 Token 数。
    """
    file_path = config["path"]

    if not file_path.exists():
        return {
            "System": config["label"],
            "Total Questions": 0,
            "Avg Input Tokens": 0,
            "Avg Output Tokens": 0,
            "Total Input Tokens": 0,
            "Total Output Tokens": 0,
            "Status": "File Not Found",
        }

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return {
            "System": config["label"],
            "Status": f"Error loading JSON: {e}",
            "Total Questions": 0,
            "Avg Input Tokens": 0,
            "Avg Output Tokens": 0,
            "Total Input Tokens": 0,
            "Total Output Tokens": 0,
        }

    total_input_tokens = 0
    total_output_tokens = 0
    valid_count = 0

    for item in data:
        input_t, output_t = get_tokens(item, config)
        if input_t > 0 or output_t > 0:
            # 仅在至少有一个 Token 数大于 0 时计入有效计数，避免统计缺失记录
            total_input_tokens += input_t
            total_output_tokens += output_t
            valid_count += 1

    return {
        "System": config["label"],
        "Total Questions": valid_count,
        "Avg Input Tokens": round(total_input_tokens / valid_count, 2)
        if valid_count > 0
        else 0,
        "Avg Output Tokens": round(total_output_tokens / valid_count, 2)
        if valid_count > 0
        else 0,
        "Total Input Tokens": total_input_tokens,
        "Total Output Tokens": total_output_tokens,
        "Status": "Success",
    }


# --- 3. 主执行逻辑 ---


def main():
    """主函数，迭代所有系统并打印结果"""
    print("--- 开始分析平均 Token 使用量 ---")

    results = []

    # 迭代配置列表，对每个系统进行分析
    for config in SYSTEM_CONFIGS:
        result = analyze_tokens(config)
        results.append(result)

    # 将结果转换为 DataFrame
    df = pd.DataFrame(results)

    # 调整列顺序，使其更具可读性
    df = df[
        [
            "System",
            "Total Questions",
            "Avg Input Tokens",
            "Avg Output Tokens",
            "Total Input Tokens",
            "Total Output Tokens",
            "Status",
        ]
    ]

    print("\n--- Token 平均使用量报告 ---")
    # 使用 to_string() 避免 'tabulate' 依赖，确保兼容性
    print(df.to_string(index=False))

    # 可选：将结果保存到 CSV 文件
    output_path = Path("./Token_Usage_Report.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n报告已保存至: {output_path}")

if __name__ == "__main__":
    main()