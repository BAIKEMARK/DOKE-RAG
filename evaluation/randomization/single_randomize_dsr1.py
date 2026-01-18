import os
import json
import csv
import asyncio
import random  # 1. 导入 random 模块
from pathlib import Path
from collections import defaultdict
from doke_rag.config.paths import ENV_FILE, RESULTS_DIR

# --- 1. 全局常量与配置 ---

# BENCHMARK_GROUP_NAME 定义了哪个组是作为衡量标准的"基准组"
BENCHMARK_GROUP_NAME = "Naive"

# 使用字典统一管理所有待评测组的文件路径
# 注意：以下路径使用相对路径，相对于项目根目录
# 请根据实际实验结果位置修改
GROUP_FILES = {
    "Naive": RESULTS_DIR / "naive_raw_checked_new_result.json",
    "StruMech": RESULTS_DIR / "split_checked_ds14_txt_new_result.json",
    "_Manual": RESULTS_DIR / "split_checked_ds14_txt_new_result.json",
    # "_Prompt": RESULTS_DIR / "split_checked_ds14_txt_new_result.json",
    "LightRAG": RESULTS_DIR / "raw_checked_new_result.json",
    "Only_Manual": RESULTS_DIR / "manual_graph_result.json",
}


# 加载环境变量（API Key等）
from dotenv import load_dotenv

load_dotenv(dotenv_path=ENV_FILE)


# --- 2. 数据加载辅助函数 ---


def extract_answer(record):
    """尝试从记录中提取答案字段"""
    if "response" in record:
        return record["response"]
    if "result" in record:
        return record["result"]
    return ""


def load_group_answers(file_path):
    """加载指定文件中的所有记录，返回 dict: { query -> answer }"""
    mapping = {}
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for rec in data:
        q = rec.get("query", "").strip()
        if q:
            mapping[q] = extract_answer(rec)
    return mapping


# --- 3. 构造评价提示词 ---


def construct_prompt(query, answer1, answer2):
    """
    构造一个通用的英文提示词，不再指定哪个是基准或挑战者。
    """
    prompt = f"""
    Role: You are an expert evaluator tasked with systematically assessing two answers to the same question based on predefined criteria.

    Goal: Compare the two answers on the criteria below, providing a specific explanation for each. Finally, determine which answer is superior overall.
    Notice that differences in language should not affect the results of your judgment.

    i) Comprehensiveness: How thoroughly does the answer address all aspects and details of the question?
    ii) Diversity: How varied and rich is the answer in offering different perspectives and insights related to the question?
    iii) Empowerment: How effectively does the answer enable the reader to understand the topic and make informed judgments?
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
    return prompt


# --- 4. API 调用与结果解析 ---


async def call_deepseek(prompt):
    """调用阿里云百炼 deepseek-r1 API 完成评价"""
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("ALIYUN_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    def sync_call():
        completion = client.chat.completions.create(
            model="deepseek-r1", messages=[{"role": "user", "content": prompt}]
        )
        return {"content": completion.choices[0].message.content, "reasoning": ""}

    result = await asyncio.to_thread(sync_call)
    return result


def parse_evaluation_result(response_text, context_info=None):
    """解析 deepseek-r1 返回的 JSON 格式评价结果，并修复特殊字符问题"""
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
        print(f"--- 解析评价结果失败: {e} ---")
        if context_info:
            print(f"问题: {context_info['question'][:100]}...")
            print(f"基准组: {context_info['benchmark_group']}")
            print(f"挑战组: {context_info['challenger_group']}")
        print("原始返回文本:", cleaned_text)
        print("---------------------------------")
        return {}


# --- 5. CSV 写入与胜率汇总函数 (已修改) ---


def write_pairwise_csv_row(
    csv_writer,
    question,
    benchmark_group,
    challenger_group,
    eval_result,
    reasoning,
    prompt,
):
    """将一对比较结果写入 CSV 行"""
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
    """统计每个挑战者组相对于基准组的胜率"""
    dims = ["Comprehensiveness", "Diversity", "Empowerment", "Overall"]
    stats = {
        group: {dim: {"wins": 0, "total": 0} for dim in dims}
        for group in all_group_names
        if group != BENCHMARK_GROUP_NAME
    }

    for record in pairwise_results:
        _, challenger_group, eval_result = record
        for dim in dims:
            # *** 修改点 ***: 直接检查获胜者是否为 "Challenger"
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
    """在所有题目上，累计每个挑战者组相对于基准组的胜率"""
    dims = ["Comprehensiveness", "Diversity", "Empowerment", "Overall"]
    global_stats = {
        group: {dim: {"wins": 0, "total": 0} for dim in dims}
        for group in all_group_names
        if group != BENCHMARK_GROUP_NAME
    }

    for question, comparisons in all_question_comparisons.items():
        for record in comparisons:
            _, challenger_group, eval_result = record
            for dim in dims:
                # *** 修改点 ***: 直接检查获胜者是否为 "Challenger"
                winner = eval_result.get(dim, {}).get("Winner", "")
                if winner:
                    global_stats[challenger_group][dim]["total"] += 1
                    if winner == "Challenger":
                        global_stats[challenger_group][dim]["wins"] += 1

    final_global_rates = {}
    for group, data in global_stats.items():
        final_global_rates[group] = {}
        for dim in dims:
            total = data[dim]["total"]
            wins = data[dim]["wins"]
            win_rate = wins / total if total > 0 else 0.0
            final_global_rates[group][dim] = win_rate

    return final_global_rates


# --- 6. 主执行流程 ---


async def main():
    # ... (数据准备部分不变) ...
    print("开始加载所有组的答案数据...")
    group_answers = {
        group_name: load_group_answers(file_path)
        for group_name, file_path in GROUP_FILES.items()
    }
    all_group_names = list(group_answers.keys())
    all_queries = list(group_answers[BENCHMARK_GROUP_NAME].keys())
    print(f"数据加载完成，共 {len(all_group_names)} 个组，{len(all_queries)} 个问题。")

    all_answers_by_query = defaultdict(dict)
    for group_name, answers in group_answers.items():
        for query, answer in answers.items():
            if query in all_queries:
                all_answers_by_query[query][group_name] = answer

    # ... (文件与目录设置部分不变) ...
    output_dir = Path("ch10_randomized_v5")  # 更新目录名
    output_dir.mkdir(exist_ok=True)
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

    # --- 并发处理 ---
    semaphore = asyncio.Semaphore(3)

    async def process_question(q, answer_dict_for_q):
        async with semaphore:
            print(f"正在评测问题: {q[:50]}...")

            async def compare_pair(challenger_group_name):
                benchmark_answer = answer_dict_for_q.get(BENCHMARK_GROUP_NAME, "")
                challenger_answer = answer_dict_for_q.get(challenger_group_name, "")

                if not benchmark_answer or not challenger_answer:
                    print(
                        f"警告：问题 '{q[:30]}...' 的组 '{challenger_group_name}' 缺少答案，跳过比较。"
                    )
                    return None

                # *** 修改点 ***: 随机打乱答案顺序
                if random.choice([True, False]):
                    answer1, answer2 = benchmark_answer, challenger_answer
                    challenger_slot = "Answer 2"
                else:
                    answer1, answer2 = challenger_answer, benchmark_answer
                    challenger_slot = "Answer 1"

                prompt = construct_prompt(q, answer1, answer2)
                deepseek_response = await call_deepseek(prompt)

                context_info = {
                    "question": q,
                    "benchmark_group": BENCHMARK_GROUP_NAME,
                    "challenger_group": challenger_group_name,
                }
                eval_result = parse_evaluation_result(
                    deepseek_response.get("content", ""), context_info
                )
                reasoning = deepseek_response.get("reasoning", "")

                if not eval_result:
                    return None

                # *** 修改点 ***: 为胜率统计创建一个"翻译"后的结果
                aggregation_result = {}
                for dim, details in eval_result.items():
                    # 安全检查，防止eval_result格式意外错误
                    if not isinstance(details, dict):
                        continue

                    winner = details.get("Winner")
                    if winner == challenger_slot:
                        aggregation_result[dim] = {"Winner": "Challenger"}
                    else:
                        aggregation_result[dim] = {"Winner": "Benchmark"}

                # 写入CSV时，仍然使用原始的 eval_result (包含Answer 1/2)
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
                    "chain_of_thought": reasoning,
                }
                # 返回时，为聚合逻辑提供翻译后的结果
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

            # 传递给聚合函数的是翻译后的结果 r[2]
            return q, [(r[0], r[1], r[2]) for r in valid_results]

    # ... (后续的汇总与写入流程基本不变，因为它现在接收的是已经处理好的结果) ...
    tasks = [
        process_question(q, ans_dict) for q, ans_dict in all_answers_by_query.items()
    ]
    question_results = await asyncio.gather(*tasks)

    print("\n所有问题评测完成，开始汇总结果...")
    all_question_comparisons = defaultdict(list)
    question_summaries = []
    for q, comparisons in question_results:
        if comparisons:
            all_question_comparisons[q].extend(comparisons)
            summary = aggregate_question_results(q, comparisons, all_group_names)
            question_summaries.append(summary)

    dims = ["Comprehensiveness", "Diversity", "Empowerment", "Overall"]
    summary_fieldnames = ["question"]
    challenger_groups = [g for g in all_group_names if g != BENCHMARK_GROUP_NAME]
    for group in challenger_groups:
        for dim in dims:
            summary_fieldnames.append(f"{group}_{dim}")

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
    global_fieldnames = ["challenger_group"] + [f"{dim}" for dim in dims]

    with open(global_summary_csv_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=global_fieldnames)
        writer.writeheader()
        for group, rates_data in global_rates.items():
            row = {"challenger_group": group}
            for dim in dims:
                rate = rates_data.get(dim, 0.0)
                row[f"{dim}"] = f"{rate:.2%}"
            writer.writerow(row)

    print(f"\n评测流程结束。结果已写入目录：'{output_dir}'")
    print(f" - 两两对比详情: {pairwise_csv_file}")
    print(f" - 单题胜率汇总: {question_summary_csv_file}")
    print(f" - 全局胜率总览: {global_summary_csv_file}")


if __name__ == "__main__":
    asyncio.run(main())