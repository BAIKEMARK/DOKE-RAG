import os
import json
import csv
import asyncio
from pathlib import Path
from collections import defaultdict

# 导入原生异步 OpenAI 客户端
from openai import AsyncOpenAI

# 导入健壮的 JSON 解析库
import demjson3
import re
from doke_rag.config.paths import ENV_FILE, RESULTS_DIR

# --- 1. 全局常量与配置 ---

# BENCHMARK_GROUP_NAME 定义了哪个组是作为衡量标准的"基准组"
BENCHMARK_GROUP_NAME = "Naive"

# 使用字典统一管理所有待评测组的文件路径
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


# --- 3. 构造评价提示词 (加强版) ---
def construct_prompt(query, benchmark_answer, challenger_answer):
    """
    构造一个带有严格规则和高质量范例的英文提示词，以确保LLM输出格式完美。
    """
    example_output = """
{
  "Comprehensiveness": {
    "Winner": "Answer 1",
    "Explanation": "Answer 1 is more comprehensive because it not only defines AI but also explains its relationship with Machine Learning and Deep Learning, which Answer 2 fails to do."
  },
  "Diversity": {
    "Winner": "Answer 1",
    "Explanation": "Answer 1 provides more diverse perspectives by mentioning different types of AI, such as Narrow AI and General AI. It also gives a historical context, which adds richness to the explanation."
  },
  "Empowerment": {
    "Winner": "Answer 2",
    "Explanation": "Answer 2 is more empowering for a beginner because it uses a simple analogy of a child learning, which makes the core concept very easy to grasp. It avoids jargon."
  },
  "Overall": {
    "Winner": "Answer 1",
    "Explanation": "Overall, Answer 1 is superior. Its comprehensive nature and diverse insights outweigh the simplicity of Answer 2. It correctly describes Machine Learning as a \\"subset\\" of AI, which is a critical detail."
  }
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
   - WRONG: {{"key": "value",}}
   - CORRECT: {{"key": "value"}}

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
Answer 1: {benchmark_answer}
Answer 2: {challenger_answer}
""".strip()
    return prompt


# --- 4. API 调用与结果解析 (最终修复版) ---


async def call_deepseek(prompt):
    """调用阿里云百炼 deepseek API，并以原生异步流式方式正确获取思考过程和最终结果。"""
    # 使用原生异步客户端
    client = AsyncOpenAI(
        api_key=os.getenv("ALIYUN_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    try:
        # 使用 await 发起原生异步流式请求
        completion_stream = await client.chat.completions.create(
            model="deepseek-v3.1",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8192,
            temperature=0.3,
            stream=True,
            extra_body={"enable_thinking": True},
        )

        reasoning_content = ""
        answer_content = ""

        # 使用 async for 进行原生异步迭代
        async for chunk in completion_stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # 收集思考过程
            if (
                hasattr(delta, "reasoning_content")
                and delta.reasoning_content is not None
            ):
                reasoning_content += delta.reasoning_content

            # 收集最终答案
            if hasattr(delta, "content") and delta.content:
                answer_content += delta.content

        return {
            "content": answer_content,
            "reasoning": reasoning_content,
        }
    except Exception as e:
        print(f"--- 调用 DeepSeek API 失败: {e} ---")
        return {"content": "", "reasoning": ""}


import re
import demjson3

def parse_evaluation_result(response_text):
    """
    使用更宽容的 demjson3 库来解析 LLM 可能生成的、格式不完美的 JSON。
    自动修复：补全缺少的大括号 / 逗号。
    """
    if not response_text:
        return {}

    # 去掉 Markdown 包裹
    cleaned_text = re.sub(
        r"^```json\s*|\s*```$", "", response_text, flags=re.MULTILINE | re.DOTALL
    ).strip()

    # 🔧 如果缺少开头的 {
    if not cleaned_text.startswith("{"):
        cleaned_text = "{" + cleaned_text

    # 🔧 如果缺少结尾的 }
    if not cleaned_text.endswith("}"):
        cleaned_text = cleaned_text + "}"

    # 🔧 如果大括号数量不平衡，补齐
    open_braces = cleaned_text.count("{")
    close_braces = cleaned_text.count("}")
    if close_braces < open_braces:
        cleaned_text += "}" * (open_braces - close_braces)

    try:
        return demjson3.decode(cleaned_text)
    except demjson3.JSONDecodeError as e:
        print(f"--- DEMJSON解析失败: {e} ---")
        print("原始返回文本:", cleaned_text)
        print("---------------------------------")
        return {}


# --- 5. CSV 写入与胜率汇总函数 (无需改动) ---


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
    for record in pairwise_results:
        _, challenger_group, eval_result = record
        for dim in dims:
            winner = eval_result.get(dim, {}).get("Winner", "")
            if winner:
                stats[challenger_group][dim]["total"] += 1
                if winner == "Answer 2":
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
        for record in comparisons:
            _, challenger_group, eval_result = record
            for dim in dims:
                winner = eval_result.get(dim, {}).get("Winner", "")
                if winner:
                    global_stats[challenger_group][dim]["total"] += 1
                    if winner == "Answer 2":
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


# --- 6. 主执行流程 (无需改动) ---


async def main():
    # 数据准备
    print("开始加载所有组的答案数据...")
    group_answers = {
        name: load_group_answers(path) for name, path in GROUP_FILES.items()
    }
    all_group_names = list(group_answers.keys())
    all_queries = list(group_answers[BENCHMARK_GROUP_NAME].keys())
    print(f"数据加载完成，共 {len(all_group_names)} 个组，{len(all_queries)} 个问题。")
    all_answers_by_query = defaultdict(dict)
    for group_name, answers in group_answers.items():
        for query, answer in answers.items():
            if query in all_queries:
                all_answers_by_query[query][group_name] = answer

    # 文件与目录设置
    output_dir = Path("dsv3_1_evaluate_stream_final_去标签")  # 更新输出目录名
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

    # 并发处理
    semaphore = asyncio.Semaphore(3)
    all_question_comparisons = defaultdict(list)

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

                prompt = construct_prompt(q, benchmark_answer, challenger_answer)

                # *** 使用最终修复版的 call_deepseek ***
                api_response = await call_deepseek(prompt)

                eval_result = parse_evaluation_result(api_response.get("content", ""))
                reasoning = api_response.get("reasoning", "")
                if not eval_result:
                    return None

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
                return (
                    BENCHMARK_GROUP_NAME,
                    challenger_group_name,
                    eval_result,
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
    print("\n所有问题评测完成，开始汇总结果...")
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
    global_fieldnames = ["challenger_group"] + dims
    with open(global_summary_csv_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=global_fieldnames)
        writer.writeheader()
        for group, rates_data in global_rates.items():
            row = {"challenger_group": group}
            for dim in dims:
                rate = rates_data.get(dim, 0.0)
                row[dim] = f"{rate:.2%}"
            writer.writerow(row)

    print(f"\n评测流程结束。结果已写入目录：'{output_dir}'")
    print(f" - 两两对比详情: {pairwise_csv_file}")
    print(f" - 单题胜率汇总: {question_summary_csv_file}")
    print(f" - 全局胜率总览: {global_summary_csv_file}")


if __name__ == "__main__":
    # 确保在运行前已安装所需库: pip install openai "python-dotenv" demjson3
    asyncio.run(main())