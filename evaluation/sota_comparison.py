# -*- coding: utf-8 -*-
import os
import json
import asyncio
import re
from pathlib import Path
import random
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
from doke_rag.config.paths import ENV_FILE, RESULTS_DIR

# --- 1. 配置区 (请根据实际情况修改) ---

# 【配置】API密钥文件路径 - 从配置模块导入
ENV_FILE_PATH = ENV_FILE

# 【配置】输出目录
OUTPUT_DIRECTORY = RESULTS_DIR / "Final_PK_Comparison_Report_5Runs"

# 【配置】并发请求数量
CONCURRENT_REQUESTS = 30

# 【新增配置】实验运行次数
NUM_EXPERIMENT_RUNS = 5

# 主角 A: DOKE-RAG (System A)
# 注意：以下路径使用相对路径，相对于项目根目录
# 请根据实际实验结果位置修改
SYSTEM_A = {
    "label": "DOKE-RAG",
    "path": Path(
        "./evaluation/results/batch_experiment/cs0.2_tk40/stru_mech_result.json"
    ),
    "answer_key": "result",
}

# 对手 B: Light RAG (System B)
SYSTEM_B = {
    "label": "LightRAG",
    "path": Path(
        "./evaluation/results/batch_experiment/cs0.2_tk80/lightrag_result.json"
    ),
    "answer_key": "result",
}

# 对手 C: Graph RAG (System C)
SYSTEM_C = {
    "label": "GraphRAG",
    "path": Path(
        "./evaluation/results/GraphRAG_local_search_result.json"
    ),
    "answer_key": "response",
}

# 评估指标列表
METRICS = ["Comprehensiveness", "Diversity", "Empowerment", "Overall"]

# --- 2. 辅助函数 ---


def construct_prompt(query: str, answer1: str, answer2: str) -> str:
    """构建评估Prompt，确保包含4个维度的定义"""
    # 提示 LLM 尽量避免平局
    return f"""
    Role: You are an expert evaluator tasked with systematically assessing two answers to the same question based on predefined criteria.
    Goal: Compare the two answers on the criteria below, providing a specific explanation for each. Finally, determine which answer is superior overall.
    Guiding Principle for Fairness: Your evaluation must weigh both the accuracy of the text and the effectiveness of any supporting materials. A high-quality answer excels in both. The ultimate measure is how effectively the entire answer conveys the necessary information and empowers the reader.
    Notice that differences in language should not affect the results of your judgment. **Your final "Winner" selection MUST be either "Answer 1" or "Answer 2". Avoid ties unless the answers are perfectly identical and equally good/bad.**
    i) Comprehensiveness: How thoroughly does the answer address all aspects of the question? For technical topics, this may include key formulas or diagrams. The focus should be on whether these components are **necessary for a complete answer and accurately presented**.
    ii) Diversity: How varied and rich is the answer in offering different perspectives and insights related to the question?
    iii) Empowerment: How effectively does the answer enable the reader to understand the topic? This is a critical measure of quality.
    - **Superiority of Good Visuals:** Under otherwise equal conditions, an answer that uses **correct, relevant, and well-explained** diagrams or formulas to clarify complex points **is superior** to a text-only answer. Such elements provide a more direct and intuitive path to understanding.
    - **Detriment of Bad Visuals:** Conversely, if an answer includes **irrelevant, incorrect, or confusing** supporting materials, it should be considered inferior to a clear and accurate text-only answer.
    iv) Overall: This dimension assesses the cumulative performance across the three preceding criteria to identify the best overall answer.
    Please strictly adhere to the following JSON format for your output. Do not include any text outside of the JSON structure.
    [Output Format]
    {{
      "Comprehensiveness": {{"Winner": "Answer 1 or Answer 2", "Explanation": "Your reasoning here"}},
      "Diversity": {{"Winner": "Answer 1 or Answer 2", "Explanation": "Your reasoning here"}},
      "Empowerment": {{"Winner": "Answer 1 or Answer 2", "Explanation": "Your reasoning here"}},
      "Overall": {{"Winner": "Answer 1 or Answer 2", "Explanation": "Your reasoning here"}}
    }}
    [Question]: {query}
    [Answers]
    Answer 1: {answer1}
    Answer 2: {answer2}
    """.strip()


async def call_deepseek(prompt: str, client: OpenAI) -> dict:
    """异步调用 LLM"""

    def sync_call():
        try:
            completion = client.chat.completions.create(
                model="deepseek-r1",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            return {"content": completion.choices[0].message.content}
        except Exception as e:
            tqdm.write(f"  - API call failed: {e}")
            return {"content": "", "error": str(e)}

    return await asyncio.to_thread(sync_call)


def parse_evaluation_result(response_text: str) -> dict:
    """解析 JSON 响应"""
    if not response_text:
        return {}
    try:
        return json.loads(response_text)
    except Exception:
        try:
            # 尝试提取 markdown 代码块或寻找首尾括号
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start != -1 and json_end != -1:
                return json.loads(response_text[json_start:json_end])
            return {}
        except Exception:
            return {}


def load_all_answers(config: dict) -> dict:
    """加载指定路径的JSON文件，并将其转换为 {query: answer} 字典。"""
    file_path = config["path"]
    label = config["label"]
    answer_key = config["answer_key"]

    if not file_path.exists():
        tqdm.write(f"错误: 未找到文件 '{label}' 路径: {file_path}")
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        tqdm.write(f"错误: 读取或解析文件 '{label}' 失败 ({file_path}): {e}")
        return None

    answer_map = {}
    for item in data:
        query = item.get("query")
        answer = item.get(answer_key)
        if query and answer:
            answer_map[query.strip()] = answer
        elif query and not answer:
            tqdm.write(
                f"警告: 文件 '{label}' 中 Query '{query[:30]}...' 缺少答案字段 '{answer_key}'。"
            )

    tqdm.write(f"成功从 '{label}' 加载了 {len(answer_map)} 个问题-答案对。")
    return answer_map


# --- 3. 核心逻辑: A vs B 和 A vs C ---


async def run_comparison(
    query_list: list,
    a_answers: dict,
    opponent_config: dict,
    client: OpenAI,
    run_id: int,
):
    """
    运行 A vs Opponent 的对比，并收集4个维度的结果
    """

    opponent_answers = load_all_answers(opponent_config)
    if not opponent_answers:
        tqdm.write(f"警告: 无法为对手 {opponent_config['label']} 找到答案，跳过对比。")
        return []

    results = []
    tasks = []
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    async def evaluate_pair(query):
        async with semaphore:
            query_key = query.strip()
            text_a = a_answers.get(query_key)
            text_op = opponent_answers.get(query_key)

            if not text_a or not text_op:
                return None

            # 随机交换位置以避免位置偏差 (Position Bias)
            is_a_first = random.choice([True, False])
            if is_a_first:
                prompt = construct_prompt(query, text_a, text_op)
                slot_a = "Answer 1"
                slot_op = "Answer 2"
            else:
                prompt = construct_prompt(query, text_op, text_a)
                slot_a = "Answer 2"
                slot_op = "Answer 1"

            api_res = await call_deepseek(prompt, client)
            eval_json = parse_evaluation_result(api_res.get("content", ""))

            row = {
                "Experiment_Run": run_id,  # 新增字段：实验轮次
                "Query": query,
                "System_A": SYSTEM_A["label"],
                "System_Opponent": opponent_config["label"],
            }

            for metric in METRICS:
                winner_str = eval_json.get(metric, {}).get("Winner", "")

                # 记录 System A 的胜负状态
                if winner_str == slot_a:
                    result_label = SYSTEM_A["label"]  # A 赢
                elif winner_str == slot_op:
                    result_label = opponent_config["label"]  # Opponent 赢
                else:
                    result_label = "Tie/Unknown"  # 平局或无法判断

                # 记录详细结果
                row[f"{metric}_Winner"] = result_label
                row[f"{metric}_Reason"] = eval_json.get(metric, {}).get(
                    "Explanation", ""
                )

            return row

    # 筛选只在 A 和 Opponent 中都存在的 Query 进行 PK
    valid_queries = [
        q
        for q in query_list
        if q.strip() in a_answers and q.strip() in opponent_answers
    ]
    tqdm.write(
        f"  - 第 {run_id} 轮：{SYSTEM_A['label']} vs {opponent_config['label']} 有 {len(valid_queries)} 个共同问题用于 PK。"
    )

    tasks = [evaluate_pair(query) for query in valid_queries]

    completed_matches = await tqdm_asyncio.gather(
        *tasks,
        desc=f"  - 第 {run_id} 轮评估 {SYSTEM_A['label']} vs {opponent_config['label']} 进度",
    )

    # 过滤掉无效结果
    results = [r for r in completed_matches if r is not None]
    return results


def generate_summary_report(all_results: list) -> pd.DataFrame:
    """
    从详细结果中生成最终的胜率统计报告，平局将从总场次中排除。
    """
    df = pd.DataFrame(all_results)
    summary_rows = []

    # 按照对手分组
    for opponent, subset_op in df.groupby("System_Opponent"):
        # 定义行标签
        row_comp = {"Metric": "Comprehensiveness"}
        row_emp = {"Metric": "Empowerment"}
        row_div = {"Metric": "Diversity"}
        row_over = {"Metric": "Overall"}

        # 统计四个指标的平均胜率
        for metric in METRICS:
            # 筛选出有明确胜负的场次 (排除平局)
            subset_metric = subset_op[
                subset_op[f"{metric}_Winner"] != "Tie/Unknown"
            ].copy()
            total_valid = len(subset_metric)  # 计入 PK 总场次的场次

            # 统计 A 赢、对手赢的数量
            wins_a = len(
                subset_metric[subset_metric[f"{metric}_Winner"] == SYSTEM_A["label"]]
            )

            if total_valid > 0:
                # 计算 A 的胜率和对手的胜率 (保证两者相加为 100%)
                win_rate_a = wins_a / total_valid
                win_rate_op = 1.0 - win_rate_a

                # 统计所有轮次中，平局的平均比例（仅作参考信息）
                total_runs = len(subset_op)
                ties = total_runs - total_valid
                avg_tie_rate = ties / total_runs
            else:
                win_rate_a = 0.0
                win_rate_op = 0.0
                avg_tie_rate = 0.0

            # 填充统计数据到对应的行字典
            if metric == "Comprehensiveness":
                row = row_comp
            elif metric == "Empowerment":
                row = row_emp
            elif metric == "Diversity":
                row = row_div
            else:  # Overall
                row = row_over

            # 以百分比显示，并加上排除平局后的有效场次信息
            row[SYSTEM_A["label"]] = f"{win_rate_a:.1%}"
            row[opponent] = f"{win_rate_op:.1%}"
            row["Total Valid Matches"] = total_valid
            row["Avg Tie Rate"] = f"{avg_tie_rate:.1%}"

        # 整理成类似图片的格式，每组对比包含 4 行
        summary_rows.append({"Opponent_Group": opponent, **row_comp})
        summary_rows.append({"Opponent_Group": opponent, **row_emp})
        summary_rows.append({"Opponent_Group": opponent, **row_div})
        summary_rows.append({"Opponent_Group": opponent, **row_over})

    # 重新构造表格，使其更像图片样式
    final_pivot_data = []
    for opponent in pd.unique([r["Opponent_Group"] for r in summary_rows]):
        final_pivot_data.append(
            {
                "Metric": "",
                SYSTEM_A["label"]: "",
                opponent: "",
                "Total Valid Matches": "",
                "Avg Tie Rate": "",
            }
        )  # 空行用于分隔
        final_pivot_data.append(
            {
                "Metric": f"--- {opponent} ---",
                SYSTEM_A["label"]: "---",
                opponent: "---",
                "Total Valid Matches": "---",
                "Avg Tie Rate": "---",
            }
        )

        # 筛选出当前对手的 4 个指标行
        subset = [r for r in summary_rows if r["Opponent_Group"] == opponent]
        for row in subset:
            final_pivot_data.append(
                {
                    "Metric": row["Metric"],
                    SYSTEM_A["label"]: row[SYSTEM_A["label"]],
                    opponent: row[opponent],
                    "Total Valid Matches": row["Total Valid Matches"],
                    "Avg Tie Rate": row["Avg Tie Rate"],
                }
            )

    # 确保列名顺序
    columns_order = [
        "Metric",
        SYSTEM_A["label"],
        opponent,
        "Total Valid Matches",
        "Avg Tie Rate",
    ]
    return pd.DataFrame(final_pivot_data)


# --- 4. 主程序 ---


async def main():
    """主执行函数"""
    load_dotenv(dotenv_path=ENV_FILE_PATH)
    api_key = os.getenv("ALIYUN_API_KEY")
    if not api_key:
        print(
            f"错误: 找不到 ALIYUN_API_KEY。请检查 .env 文件 (路径: {ENV_FILE_PATH}) 或环境变量设置。"
        )
        return

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    output_dir = Path(OUTPUT_DIRECTORY)
    output_dir.mkdir(exist_ok=True, parents=True)

    # 1. 加载所有系统的答案
    tqdm.write("\n--- 1. 加载所有系统答案 (只加载一次) ---")
    a_answers = load_all_answers(SYSTEM_A)
    if not a_answers:
        return

    unique_queries = sorted(a_answers.keys())
    tqdm.write(f"以 {SYSTEM_A['label']} 为基准，共找到 {len(unique_queries)} 个问题。")

    # 2. 运行 N 轮 PK 对比
    tqdm.write(f"\n--- 2. 运行 {NUM_EXPERIMENT_RUNS} 轮 PK 对比: A vs B & A vs C ---")
    opponents = [SYSTEM_B, SYSTEM_C]
    all_results = []

    # 外层循环：重复 N 次实验
    for run_id in tqdm(range(1, NUM_EXPERIMENT_RUNS + 1), desc="所有实验轮次"):
        for opponent in opponents:
            tqdm.write(
                f"\n{'=' * 20} 第 {run_id}/{NUM_EXPERIMENT_RUNS} 轮对比: {SYSTEM_A['label']} vs {opponent['label']} {'=' * 20}"
            )

            # run_comparison 内部会随机交换位置
            comparison_results = await run_comparison(
                unique_queries, a_answers, opponent, client, run_id
            )
            all_results.extend(comparison_results)

    # 3. 结果处理与保存
    if not all_results:
        print("\n评估未能生成任何有效结果。")
        return

    # 保存详细 CSV 报告 (包含所有 5 轮的结果)
    df_detail = pd.DataFrame(all_results)
    detail_csv = output_dir / "Detailed_Comparison_5Runs.csv"
    df_detail.to_csv(detail_csv, index=False, encoding="utf-8-sig")
    print(f"\n🎉 详细对比结果 ({NUM_EXPERIMENT_RUNS} 轮) 已保存至: {detail_csv}")

    # 生成最终统计报告 (平均结果)
    df_summary_pivot = generate_summary_report(all_results)
    summary_csv = output_dir / "Summary_WinRates_Average_Report.csv"
    df_summary_pivot.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print("\n" + "="*70)
    print(f"  📊 最终统计报告 ({NUM_EXPERIMENT_RUNS} 轮平均胜率，已排除平局)")
    print("="*70)

    # 解决打印错误: 使用 df.to_string() 替换 df.to_markdown()
    print(df_summary_pivot.to_string(index=False))

if __name__ == "__main__":
    asyncio.run(main())