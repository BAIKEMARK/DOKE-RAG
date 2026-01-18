# -*- coding: utf-8 -*-

"""
全自动批量评估脚本 (最终优化版 V2)
================================

新功能:
- `--total-batches` 和 `--current-batch` 参数，可将任务分批运行。
- 根据API限制，将默认并发数 (Semaphore) 提高到50以加速评估。
"""

import os
import json
import csv
import asyncio
import random
import argparse
import math
from pathlib import Path
from collections import defaultdict
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
from doke_rag.config.paths import ENV_FILE

# --- 1. 全局常量与配置 ---
BENCHMARK_GROUP_NAME = "Naive"
EVALUATION_DIMS = ["Comprehensiveness", "Diversity", "Empowerment", "Overall"]
F_NAMES_MAP = {
    "StruMech": "stru_mech",
    "_Manual": "_manual",
    "_Split": "_split",
    "LightRAG": "lightrag",
    "Naive": "naive",
    "Only_Manual": "only_manual",
}
CONCURRENT_REQUESTS = 40  # 优化：并发数提高到50

# --- 2. 函数定义区 ---


def extract_answer(record):
    if "response" in record:
        return record["response"]
    if "result" in record:
        return record["result"]
    return ""


def load_group_answers(file_path):
    if not Path(file_path).exists():
        tqdm.write(f"    - 警告: 结果文件未找到，将跳过: {file_path.name}")
        return {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        mapping = {}
        for rec in data:
            q = rec.get("query", "").strip()
            if q:
                mapping[q] = extract_answer(rec)
        return mapping
    except (json.JSONDecodeError, IOError) as e:
        tqdm.write(f"    - 错误: 无法读取或解析文件 {file_path.name}: {e}")
        return {}


def construct_prompt(query, answer1, answer2):
    return f"""
    Role: You are an expert evaluator tasked with systematically assessing two answers to the same question based on predefined criteria.
    Goal: Compare the two answers on the criteria below, providing a specific explanation for each. Finally, determine which answer is superior overall.
    Guiding Principle for Fairness: Your evaluation must weigh both the accuracy of the text and the effectiveness of any supporting materials. A high-quality answer excels in both. The ultimate measure is how effectively the entire answer conveys the necessary information and empowers the reader.
    Notice that differences in language should not affect the results of your judgment.
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


async def call_deepseek(prompt: str, client: OpenAI):
    def sync_call():
        completion = client.chat.completions.create(
            model="deepseek-r1", messages=[{"role": "user", "content": prompt}]
        )
        return {"content": completion.choices[0].message.content, "reasoning": ""}

    return await asyncio.to_thread(sync_call)


def parse_evaluation_result(response_text, context_info=None):
    if not response_text:
        return {}
    lines = response_text.strip().splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    cleaned_text = "\n".join(lines).strip()
    fixed_text = cleaned_text.replace("\\", "\\\\")
    try:
        return json.loads(fixed_text)
    except Exception as e:
        tqdm.write(f"    --- 解析评价结果失败: {e} ---")
        if context_info:
            tqdm.write(
                f"    Trial: {context_info.get('trial', 'N/A')}, Question: {context_info['question'][:50]}..."
            )
        return {}


def aggregate_global_results(all_question_comparisons, all_group_names):
    global_stats = {
        group: {dim: {"wins": 0, "total": 0} for dim in EVALUATION_DIMS}
        for group in all_group_names
        if group != BENCHMARK_GROUP_NAME
    }
    for _, comparisons in all_question_comparisons.items():
        for _, challenger_group, eval_result in comparisons:
            for dim in EVALUATION_DIMS:
                winner = eval_result.get(dim, {}).get("Winner", "")
                if winner:
                    global_stats[challenger_group][dim]["total"] += 1
                    if winner == "Challenger":
                        global_stats[challenger_group][dim]["wins"] += 1
    final_rates = {}
    for group, data in global_stats.items():
        final_rates[group] = {
            dim: (
                data[dim]["wins"] / data[dim]["total"]
                if data[dim]["total"] > 0
                else 0.0
            )
            for dim in EVALUATION_DIMS
        }
    return final_rates


# --- 3. 核心评估逻辑区 ---
async def run_single_trial(
    trial_number: int, group_files: dict, base_output_dir: Path, client: OpenAI
):
    group_answers = {
        name: load_group_answers(path) for name, path in group_files.items()
    }
    all_group_names = list(group_answers.keys())

    if (
        BENCHMARK_GROUP_NAME not in group_answers
        or not group_answers[BENCHMARK_GROUP_NAME]
    ):
        tqdm.write(
            f"    - 错误: 基准组 '{BENCHMARK_GROUP_NAME}' 数据为空或未找到，无法进行比较。"
        )
        return None

    all_queries = list(group_answers[BENCHMARK_GROUP_NAME].keys())
    all_answers_by_query = defaultdict(dict)
    for group_name, answers in group_answers.items():
        for query, answer in answers.items():
            if query in all_queries:
                all_answers_by_query[query][group_name] = answer

    trial_output_dir = base_output_dir / f"trial_{trial_number}"
    trial_output_dir.mkdir(exist_ok=True, parents=True)
    pairwise_csv_file = trial_output_dir / "pairwise_comparisons.csv"

    pairwise_fieldnames = (
        ["question", "benchmark_group", "challenger_group", "prompt"]
        + [
            f"{dim}_{suffix}"
            for dim in EVALUATION_DIMS
            for suffix in ["winner", "reasoning"]
        ]
        + ["chain_of_thought"]
    )

    with open(pairwise_csv_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pairwise_fieldnames)
        writer.writeheader()

    all_question_comparisons = defaultdict(list)
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    async def process_question(q, answer_dict_for_q):
        async with semaphore:

            async def compare_pair(challenger_group):
                benchmark_answer = answer_dict_for_q.get(BENCHMARK_GROUP_NAME, "")
                challenger_answer = answer_dict_for_q.get(challenger_group, "")
                if not benchmark_answer or not challenger_answer:
                    return None

                answer1, answer2, challenger_slot = (
                    (benchmark_answer, challenger_answer, "Answer 2")
                    if random.choice([True, False])
                    else (challenger_answer, benchmark_answer, "Answer 1")
                )
                prompt = construct_prompt(q, answer1, answer2)
                api_response = await call_deepseek(prompt, client)
                eval_result = parse_evaluation_result(
                    api_response.get("content", ""),
                    {"question": q, "trial": trial_number},
                )
                if not eval_result:
                    return None

                agg_result = {
                    dim: {
                        "Winner": "Challenger"
                        if details.get("Winner") == challenger_slot
                        else "Benchmark"
                    }
                    for dim, details in eval_result.items()
                    if isinstance(details, dict)
                }

                row_data = {
                    "question": q,
                    "benchmark_group": BENCHMARK_GROUP_NAME,
                    "challenger_group": challenger_group,
                    "prompt": prompt,
                    "chain_of_thought": api_response.get("reasoning", ""),
                }
                for dim in EVALUATION_DIMS:
                    details = eval_result.get(dim, {})
                    row_data[f"{dim}_winner"] = details.get("Winner", "")
                    row_data[f"{dim}_reasoning"] = details.get("Explanation", "")

                return (BENCHMARK_GROUP_NAME, challenger_group, agg_result, row_data)

            tasks = [
                compare_pair(group)
                for group in all_group_names
                if group != BENCHMARK_GROUP_NAME
            ]
            results = await asyncio.gather(*tasks)
            valid_results = [r for r in results if r]

            with open(pairwise_csv_file, "a", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=pairwise_fieldnames)
                for res_tuple in valid_results:
                    writer.writerow(res_tuple[3])
            return q, [(r[0], r[1], r[2]) for r in valid_results]

    tasks = [
        process_question(q, ans_dict) for q, ans_dict in all_answers_by_query.items()
    ]
    question_results = await tqdm_asyncio.gather(
        *tasks, desc=f"  - [Trial {trial_number}] 评测问题中", total=len(tasks)
    )

    for q, comparisons in question_results:
        if comparisons:
            all_question_comparisons[q].extend(comparisons)

    global_rates = aggregate_global_results(all_question_comparisons, all_group_names)
    global_summary_csv_file = trial_output_dir / "win_rates_summary.csv"
    with open(global_summary_csv_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["challenger_group"] + EVALUATION_DIMS)
        writer.writeheader()
        for group, rates_data in global_rates.items():
            row = {
                "challenger_group": group,
                **{dim: f"{rate:.2%}" for dim, rate in rates_data.items()},
            }
            writer.writerow(row)

    return global_summary_csv_file


async def run_evaluation_for_directory(
    input_dir: Path, eval_root_dir: Path, num_trials: int, client: OpenAI
):
    group_files = {
        key: input_dir / f"{fname}_result.json" for key, fname in F_NAMES_MAP.items()
    }
    eval_output_dir = eval_root_dir / input_dir.name
    eval_output_dir.mkdir(exist_ok=True, parents=True)
    tqdm.write(f"  评估结果将保存在: {eval_output_dir}")

    summary_files = []
    for i in tqdm(range(1, num_trials + 1), desc="  - 评估轮次进度", leave=False):
        summary_file = await run_single_trial(i, group_files, eval_output_dir, client)
        if summary_file:
            summary_files.append(summary_file)

    if not summary_files:
        tqdm.write(f"  - 评估目录 {input_dir.name} 未生成任何有效的摘要文件。")
        return

    all_trials_data = [pd.read_csv(file) for file in summary_files if file.exists()]
    if not all_trials_data:
        tqdm.write(f"  - 未找到有效的摘要文件进行最终统计。")
        return

    combined_df = pd.concat(all_trials_data, ignore_index=True)
    for dim in EVALUATION_DIMS:
        if dim in combined_df.columns:
            combined_df[dim] = combined_df[dim].str.rstrip("%").astype(float) / 100.0

    final_results = []
    for group in combined_df["challenger_group"].unique():
        for dim in EVALUATION_DIMS:
            series = combined_df[combined_df["challenger_group"] == group][dim]
            stats = {
                "challenger_group": group,
                "metric": dim,
                "mean": series.mean(),
                "variance": series.var(),
                "std_dev": series.std(),
            }
            for i, val in enumerate(series):
                stats[f"trial_{i + 1}"] = val
            final_results.append(stats)

    final_summary_df = pd.DataFrame(final_results)
    percent_cols = ["mean", "std_dev"] + [
        f"trial_{i + 1}" for i in range(len(all_trials_data))
    ]
    for col in percent_cols:
        if col in final_summary_df.columns:
            final_summary_df[col] = final_summary_df[col].apply(lambda x: f"{x:.2%}")
    if "variance" in final_summary_df.columns:
        final_summary_df["variance"] = final_summary_df["variance"].apply(
            lambda x: f"{x:.6f}"
        )

    final_summary_path = eval_output_dir / "final_statistical_summary.csv"
    final_summary_df.to_csv(final_summary_path, index=False, encoding="utf-8-sig")
    tqdm.write(f"  - 最终统计报告已保存至: {final_summary_path}")


# --- 4. 顶层控制器 ---
async def main(args):
    load_dotenv(dotenv_path=args.env_path if args.env_path else None)
    if not os.getenv("ALIYUN_API_KEY"):
        print("错误: 找不到 ALIYUN_API_KEY。请检查 .env 文件或环境变量设置。")
        return

    results_root = Path(args.results_root_dir)
    if not results_root.is_dir():
        print(f"错误: 结果根目录不存在: {results_root}")
        return

    subdirs_to_evaluate = sorted(
        [
            d
            for d in results_root.iterdir()
            if d.is_dir() and d.name.startswith("cs") and "_tk" in d.name
        ]
    )

    if args.total_batches > 0:
        if not (1 <= args.current_batch <= args.total_batches):
            print(f"错误: --current-batch 的值必须在 1 到 {args.total_batches} 之间。")
            return

        total_dirs = len(subdirs_to_evaluate)
        dirs_per_batch = math.ceil(total_dirs / args.total_batches)
        start_index = (args.current_batch - 1) * dirs_per_batch
        end_index = start_index + dirs_per_batch
        dirs_to_process = subdirs_to_evaluate[start_index:end_index]

        print(
            f"任务共分为 {args.total_batches} 批。本次运行第 {args.current_batch} 批，包含 {len(dirs_to_process)} 个目录。"
        )
    else:
        dirs_to_process = subdirs_to_evaluate

    if not dirs_to_process:
        print("当前批次没有需要处理的目录。")
        return

    print(
        f"将对以下 {len(dirs_to_process)} 个目录进行评估: {[d.name for d in dirs_to_process]}"
    )

    client = OpenAI(
        api_key=os.getenv("ALIYUN_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    for directory in tqdm(
        dirs_to_process, desc="总评估进度", total=len(dirs_to_process)
    ):
        tqdm.write(f"\n{'=' * 30} 开始处理目录: {directory.name} {'=' * 30}")
        await run_evaluation_for_directory(
            directory, Path(args.eval_root_dir), args.num_trials, client
        )

    print(f"\n{'🎉' * 10} 所有评估任务已全部完成！ {'🎉' * 10}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对多个实验结果目录进行批量评估。")
    parser.add_argument(
        "--results_root_dir",
        type=str,
        required=True,
        help="包含所有 'cs..._tk...' 结果子目录的根文件夹。",
    )
    parser.add_argument(
        "--eval_root_dir",
        type=str,
        required=True,
        help="用于保存所有评估结果的总文件夹。",
    )
    parser.add_argument(
        "--num_trials", type=int, default=5, help="对每个目录进行的独立评估轮数。"
    )
    parser.add_argument(
        "--env_path",
        type=str,
        default=str(ENV_FILE),
        help="指向 .env 文件的可选路径。",
    )
    parser.add_argument('--total-batches', type=int, default=0, help="将所有目录分成几批来运行。默认为0，即不分批，一次性运行所有。")
    parser.add_argument('--current-batch', type=int, default=1, help="当使用 --total-batches 时，指定当前要运行的是第几批（从1开始）。")

    args = parser.parse_args()
    asyncio.run(main(args))