import os
import json
import csv
import asyncio
import random
from pathlib import Path
from collections import defaultdict
import pandas as pd  # 导入 pandas 用于最终的统计分析
import numpy as np
from doke_rag.config.paths import ENV_FILE, RESULTS_DIR

# 导入原生异步 OpenAI 客户端
from openai import AsyncOpenAI

# 导入健壮的 JSON 解析库
import demjson3
import re

# --- 1. 全局常量与配置 ---

# ==============================================================================
#                                 主要配置
# ==============================================================================
# 设置您想要进行的独立实验次数
NUM_TRIALS = 5
# 总的输出目录
BASE_OUTPUT_DIR = RESULTS_DIR / "Meta_Analysis_Results"
# ==============================================================================

BENCHMARK_GROUP_NAME = "Naive"

# 注意：以下路径使用相对路径，相对于项目根目录
# 请根据实际实验结果位置修改
GROUP_FILES = {
    "Naive": RESULTS_DIR / "ds_14b_chapter10/naive_result.json",
    "StruMech": RESULTS_DIR / "ds_14b_chapter10/stru_mech_result.json",
    "_Manual": RESULTS_DIR / "ds_14b_chapter10/_manual_result.json",
    "_Prompt": RESULTS_DIR / "ds_14b_chapter10/_prompt_result.json",
    "LightRAG": RESULTS_DIR / "ds_14b_chapter10/lightrag_result.json",
    "Only_Manual": RESULTS_DIR / "ds_14b_chapter10/only_manual_result.json",
}

from dotenv import load_dotenv

load_dotenv(dotenv_path=ENV_FILE)


# --- 数据加载、提示词构造、API调用与解析 (这部分函数与上一版相同) ---


def extract_answer(record):
    if "response" in record:
        return record["response"]
    if "result" in record:
        return record["result"]
    return ""


def load_group_answers(file_path):
    mapping = {}
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for rec in data:
        q = rec.get("query", "").strip()
        if q:
            mapping[q] = extract_answer(rec)
    return mapping


def construct_prompt(query, answer1, answer2):
    example_output = """
{
  "Comprehensiveness": { "Winner": "Answer 1", "Explanation": "..." },
  "Diversity": { "Winner": "Answer 1", "Explanation": "..." },
  "Empowerment": { "Winner": "Answer 2", "Explanation": "..." },
  "Overall": { "Winner": "Answer 1", "Explanation": "..." }
}
"""
    prompt = f"""
Role: You are an expert evaluator and a meticulous JSON formatter. Your task is to systematically assess two answers to the same question based on predefined criteria and output a PERFECTLY formatted JSON object.
Goal: Compare the two answers on the criteria below, providing a specific explanation for each. Finally, determine which answer is superior overall.
[Evaluation Criteria]
i) Comprehensiveness: How thoroughly does the answer address all aspects and details of the question?
ii) Diversity: How varied and rich is the answer in offering different perspectives and insights related to the question?
iii) Empowerment: How effectively does the answer enable the reader to understand the topic and make informed judgments?
iv) Overall: This dimension assesses the cumulative performance across the three preceding criteria to identify the best overall answer.
[Strict Formatting Rules]
1. Your entire output MUST be a single JSON object. Do not include any text, notes, or apologies before or after the JSON structure.
2. Inside the JSON, all strings MUST be enclosed in double quotes.
3. If an "Explanation" string itself contains a double quote, it MUST be escaped with a backslash (e.g., "a \\"quoted\\" word").
4. There MUST NOT be a trailing comma after the last element in any JSON object or list.
[Good Example of a Perfect Output]
{example_output}
[Output Format Template]
{{
  "Comprehensiveness": {{"Winner": "Answer 1 or Answer 2", "Explanation": "Your reasoning here"}},
  "Diversity": {{"Winner": "Answer 1 or Answer 2", "Explanation": "Your reasoning here"}},
  "Empowerment": {{"Winner": "Answer 1 or Answer 2", "Explanation": "Your reasoning here"}},
  "Overall": {{"Winner": "Answer 1 or Answer 2", "Explanation": "Your reasoning here"}}
}}
[Task]
Now, evaluate the following question and answers according to all the rules and examples above.
[Question]: {query}
[Answers]
Answer 1: {answer1}
Answer 2: {answer2}
""".strip()
    return prompt


async def call_deepseek(prompt):
    client = AsyncOpenAI(
        api_key=os.getenv("ALIYUN_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    try:
        completion_stream = await client.chat.completions.create(
            model="deepseek-v3.1",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8192,
            temperature=0.3,
            stream=True,
            extra_body={"enable_thinking": True},
        )
        reasoning_content, answer_content = "", ""
        async for chunk in completion_stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                reasoning_content += delta.reasoning_content
            if hasattr(delta, "content") and delta.content:
                answer_content += delta.content
        return {"content": answer_content, "reasoning": reasoning_content}
    except Exception as e:
        print(f"--- 调用 DeepSeek API 失败: {e} ---")
        return {"content": "", "reasoning": ""}


def parse_evaluation_result(response_text):
    if not response_text:
        return {}
    cleaned_text = re.sub(
        r"^```json\s*|\s*```$", "", response_text, flags=re.MULTILINE | re.DOTALL
    ).strip()
    if not cleaned_text.startswith("{"):
        cleaned_text = "{" + cleaned_text
    if not cleaned_text.endswith("}"):
        cleaned_text = cleaned_text + "}"
    open_braces, close_braces = cleaned_text.count("{"), cleaned_text.count("}")
    if close_braces < open_braces:
        cleaned_text += "}" * (open_braces - close_braces)
    try:
        return demjson3.decode(cleaned_text)
    except demjson3.JSONDecodeError as e:
        print(
            f"--- DEMJSON解析失败: {e} ---\n原始返回文本: {cleaned_text}\n---------------------------------"
        )
        return {}


# --- CSV 写入与胜率汇总函数 ---
# (这部分函数与上一版相同)
def write_pairwise_csv_row(
    csv_writer,
    question,
    benchmark_group,
    challenger_group,
    eval_result,
    reasoning,
    prompt,
):
    row = {
        "question": question,
        "benchmark_group": benchmark_group,
        "challenger_group": challenger_group,
        "prompt": prompt,
        "Comprehensiveness_winner": eval_result.get("Comprehensiveness", {}).get(
            "Winner", ""
        ),
        "Comprehensiveness_reasoning": eval_result.get("Comprehensiveness", {}).get(
            "Explanation", ""
        ),
        "Diversity_winner": eval_result.get("Diversity", {}).get("Winner", ""),
        "Diversity_reasoning": eval_result.get("Diversity", {}).get("Explanation", ""),
        "Empowerment_winner": eval_result.get("Empowerment", {}).get("Winner", ""),
        "Empowerment_reasoning": eval_result.get("Empowerment", {}).get(
            "Explanation", ""
        ),
        "Overall_winner": eval_result.get("Overall", {}).get("Winner", ""),
        "Overall_reasoning": eval_result.get("Overall", {}).get("Explanation", ""),
        "chain_of_thought": reasoning,
    }
    csv_writer.writerow(row)


def aggregate_question_results(question, pairwise_results, all_group_names):
    dims = ["Comprehensiveness", "Diversity", "Empowerment", "Overall"]
    stats = {
        group: {dim: {"wins": 0, "total": 0} for dim in dims}
        for group in all_group_names
        if group != BENCHMARK_GROUP_NAME
    }
    for _, challenger_group, eval_result in pairwise_results:
        for dim in dims:
            winner = eval_result.get(dim, {}).get("Winner", "")
            if winner:
                stats[challenger_group][dim]["total"] += 1
                if winner == "Challenger":
                    stats[challenger_group][dim]["wins"] += 1
    win_rates = {
        group: {
            dim: (
                stats[group][dim]["wins"] / stats[group][dim]["total"]
                if stats[group][dim]["total"] > 0
                else 0.0
            )
            for dim in dims
        }
        for group in stats
    }
    return {"question": question, "win_rates": win_rates}


def aggregate_global_results(all_question_comparisons, all_group_names):
    dims = ["Comprehensiveness", "Diversity", "Empowerment", "Overall"]
    global_stats = {
        group: {dim: {"wins": 0, "total": 0} for dim in dims}
        for group in all_group_names
        if group != BENCHMARK_GROUP_NAME
    }
    for _, comparisons in all_question_comparisons.items():
        for _, challenger_group, eval_result in comparisons:
            for dim in dims:
                winner = eval_result.get(dim, {}).get("Winner", "")
                if winner:
                    global_stats[challenger_group][dim]["total"] += 1
                    if winner == "Challenger":
                        global_stats[challenger_group][dim]["wins"] += 1
    final_global_rates = {}
    for group, data in global_stats.items():
        final_global_rates[group] = {}
        for dim in dims:
            total, wins = data[dim]["total"], data[dim]["wins"]
            win_rate = wins / total if total > 0 else 0.0
            final_global_rates[group][dim] = win_rate
    return final_global_rates


# --- 2. 核心执行逻辑：单次实验 ---


async def run_single_trial(trial_number: int):
    """
    执行一次完整的评估流程，并将结果保存在指定编号的文件夹中。
    """
    print(f"\n{'=' * 25} 开始第 {trial_number} 组实验 {'=' * 25}")

    # 数据准备
    group_answers = {
        name: load_group_answers(path) for name, path in GROUP_FILES.items()
    }
    all_group_names = list(group_answers.keys())
    all_queries = list(group_answers[BENCHMARK_GROUP_NAME].keys())
    all_answers_by_query = defaultdict(dict)
    for group_name, answers in group_answers.items():
        for query, answer in answers.items():
            if query in all_queries:
                all_answers_by_query[query][group_name] = answer

    # 为本次实验创建独立的输出目录
    output_dir = BASE_OUTPUT_DIR / f"trial_{trial_number}"
    output_dir.mkdir(exist_ok=True, parents=True)

    pairwise_csv_file = output_dir / "pairwise_comparisons.csv"
    question_summary_csv_file = output_dir / "win_rates_per_question.csv"
    global_summary_csv_file = output_dir / "win_rates_summary.csv"

    pairwise_fieldnames = [
        "question",
        "benchmark_group",
        "challenger_group",
        "prompt",
        "Comprehensiveness_winner",
        "Comprehensiveness_reasoning",
        "Diversity_winner",
        "Diversity_reasoning",
        "Empowerment_winner",
        "Empowerment_reasoning",
        "Overall_winner",
        "Overall_reasoning",
        "chain_of_thought",
    ]
    with open(pairwise_csv_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pairwise_fieldnames)
        writer.writeheader()

    # 并发处理
    semaphore = asyncio.Semaphore(3)

    async def process_question(q, answer_dict_for_q):
        async with semaphore:
            print(f"  [Trial {trial_number}] 正在评测问题: {q[:50]}...")

            async def compare_pair(challenger_group_name):
                benchmark_answer = answer_dict_for_q.get(BENCHMARK_GROUP_NAME, "")
                challenger_answer = answer_dict_for_q.get(challenger_group_name, "")
                if not benchmark_answer or not challenger_answer:
                    return None

                if random.choice([True, False]):
                    answer1, answer2, challenger_slot = (
                        benchmark_answer,
                        challenger_answer,
                        "Answer 2",
                    )
                else:
                    answer1, answer2, challenger_slot = (
                        challenger_answer,
                        benchmark_answer,
                        "Answer 1",
                    )

                prompt = construct_prompt(q, answer1, answer2)
                api_response = await call_deepseek(prompt)
                eval_result = parse_evaluation_result(api_response.get("content", ""))
                if not eval_result:
                    return None

                aggregation_result = {}
                for dim, details in eval_result.items():
                    if isinstance(details, dict):
                        winner = details.get("Winner")
                        aggregation_result[dim] = {
                            "Winner": "Challenger"
                            if winner == challenger_slot
                            else "Benchmark"
                        }

                row_data = {
                    "question": q,
                    "benchmark_group": BENCHMARK_GROUP_NAME,
                    "challenger_group": challenger_group_name,
                    "prompt": prompt,
                    "Comprehensiveness_winner": eval_result.get(
                        "Comprehensiveness", {}
                    ).get("Winner", ""),
                    "Comprehensiveness_reasoning": eval_result.get(
                        "Comprehensiveness", {}
                    ).get("Explanation", ""),
                    "Diversity_winner": eval_result.get("Diversity", {}).get(
                        "Winner", ""
                    ),
                    "Diversity_reasoning": eval_result.get("Diversity", {}).get(
                        "Explanation", ""
                    ),
                    "Empowerment_winner": eval_result.get("Empowerment", {}).get(
                        "Winner", ""
                    ),
                    "Empowerment_reasoning": eval_result.get("Empowerment", {}).get(
                        "Explanation", ""
                    ),
                    "Overall_winner": eval_result.get("Overall", {}).get("Winner", ""),
                    "Overall_reasoning": eval_result.get("Overall", {}).get(
                        "Explanation", ""
                    ),
                    "chain_of_thought": api_response.get("reasoning", ""),
                }
                return (
                    BENCHMARK_GROUP_NAME,
                    challenger_group_name,
                    aggregation_result,
                    row_data,
                )

            tasks = [
                compare_pair(group)
                for group in all_group_names
                if group != BENCHMARK_GROUP_NAME
            ]
            results = await asyncio.gather(*tasks)
            valid_results = [r for r in results if r is not None]

            with open(pairwise_csv_file, "a", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=pairwise_fieldnames)
                for _, _, _, row in valid_results:
                    writer.writerow(row)
            return q, [(r[0], r[1], r[2]) for r in valid_results]

    tasks = [
        process_question(q, ans_dict) for q, ans_dict in all_answers_by_query.items()
    ]
    question_results = await asyncio.gather(*tasks)

    # 结果汇总与写入
    all_question_comparisons = defaultdict(list)
    question_summaries = []
    for q, comparisons in question_results:
        if comparisons:
            all_question_comparisons[q].extend(comparisons)
            summary = aggregate_question_results(q, comparisons, all_group_names)
            question_summaries.append(summary)

    dims = ["Comprehensiveness", "Diversity", "Empowerment", "Overall"]
    summary_fieldnames = ["question"] + [
        f"{group}_{dim}"
        for group in [g for g in all_group_names if g != BENCHMARK_GROUP_NAME]
        for dim in dims
    ]
    with open(question_summary_csv_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        for summary in question_summaries:
            row = {"question": summary["question"]}
            for group, rates in summary["win_rates"].items():
                for dim, rate in rates.items():
                    row[f"{group}_{dim}"] = f"{rate:.2%}"
            writer.writerow(row)

    global_rates = aggregate_global_results(all_question_comparisons, all_group_names)
    global_fieldnames = ["challenger_group"] + dims
    with open(global_summary_csv_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=global_fieldnames)
        writer.writeheader()
        for group, rates_data in global_rates.items():
            row = {"challenger_group": group}
            for dim in dims:
                row[dim] = f"{rates_data.get(dim, 0.0):.2%}"
            writer.writerow(row)

    print(f"--- 第 {trial_number} 组实验完成 ---")
    return global_summary_csv_file


# --- 3. 顶层控制与统计分析 ---


async def run_all_trials():
    """
    循环执行所有实验，并在最后进行统计分析。
    """
    summary_files = []
    for i in range(1, NUM_TRIALS + 1):
        summary_file = await run_single_trial(i)
        summary_files.append(summary_file)

    print(f"\n{'=' * 25} 所有 {NUM_TRIALS} 组实验已完成 {'=' * 25}")
    print("开始进行最终的统计分析...")

    all_trials_data = []
    for i, file_path in enumerate(summary_files):
        try:
            df = pd.read_csv(file_path)
            df["trial"] = i + 1
            all_trials_data.append(df)
        except FileNotFoundError:
            print(f"警告：找不到文件 {file_path}，跳过此实验的统计。")

    if not all_trials_data:
        print("错误：没有任何有效的实验结果可供分析。")
        return

    # 合并所有实验数据
    combined_df = pd.concat(all_trials_data, ignore_index=True)

    # 数据清洗：将百分比字符串转换为浮点数
    dims = ["Comprehensiveness", "Diversity", "Empowerment", "Overall"]
    for dim in dims:
        if dim in combined_df.columns:
            combined_df[dim] = combined_df[dim].str.rstrip("%").astype(float) / 100.0

    # 按挑战者组和评估维度进行分组
    grouped = combined_df.groupby(["challenger_group", pd.Grouper(key="trial")])

    # 准备存储最终结果的列表
    final_results = []

    # 获取所有挑战者组和维度
    challenger_groups = combined_df["challenger_group"].unique()

    for group in challenger_groups:
        for dim in dims:
            # 提取特定组和维度的所有实验数据
            series = combined_df[(combined_df["challenger_group"] == group)][dim]

            stats = {
                "challenger_group": group,
                "metric": dim,
                "mean": series.mean(),
                "variance": series.var(),
                "std_dev": series.std(),
            }
            # 添加每一轮的原始数据，以查看偏离情况
            for i, val in enumerate(series):
                stats[f"trial_{i + 1}"] = val

            final_results.append(stats)

    # 创建最终的统计DataFrame
    final_summary_df = pd.DataFrame(final_results)

    # 格式化输出为百分比
    percent_cols = ["mean", "std_dev"] + [
        f"trial_{i + 1}" for i in range(len(all_trials_data))
    ]
    for col in percent_cols:
        if col in final_summary_df.columns:
            final_summary_df[col] = final_summary_df[col].apply(lambda x: f"{x:.2%}")

    # variance 不应格式化为百分比
    if "variance" in final_summary_df.columns:
        final_summary_df["variance"] = final_summary_df["variance"].apply(
            lambda x: f"{x:.6f}"
        )

    # 保存最终统计报告
    final_summary_path = BASE_OUTPUT_DIR / "final_statistical_summary.csv"
    final_summary_df.to_csv(final_summary_path, index=False, encoding='utf-8-sig')

    print("\n统计分析完成！")
    print(f" - 每组实验的详细结果保存在 '{BASE_OUTPUT_DIR}/trial_*/' 目录下。")
    print(f" - 最终的统计汇总报告已保存至: {final_summary_path}")


if __name__ == "__main__":
    asyncio.run(run_all_trials())