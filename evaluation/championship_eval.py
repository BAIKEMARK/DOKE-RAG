# -*- coding: utf-8 -*-
import os
import json
import asyncio
import itertools
import re
from pathlib import Path
from collections import defaultdict
import random
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
from doke_rag.config.paths import ENV_FILE, RESULTS_DIR, EXCEL_RESULTS

# --- 1. 配置区 (您需要在这里修改路径) ---

# 【修改这里】指向 Consolidated_Report.xlsx 文件的路径 - 从配置模块导入
EXCEL_FILE_PATH = EXCEL_RESULTS

# 【修改这里】指向包含所有 'cs..._tk...' 原始JSON结果子目录的根文件夹
RESULTS_ROOT_DIR = RESULTS_DIR / "batch_experiment"

# 【修改这里】指向你的 .env 文件，用于读取API密钥 - 从配置模块导入
ENV_FILE_PATH = ENV_FILE

# 【修改这里】用于保存最终排名报告的文件夹
OUTPUT_DIRECTORY = RESULTS_DIR / "Championship_Evaluation_Final"

# --- 其他配置 ---
CONCURRENT_REQUESTS = 30
TARGET_METRIC = "Overall"

# --- 2. 辅助函数 ---


def load_data_from_excel(filepath: str) -> pd.DataFrame:
    try:
        df_raw = pd.read_excel(filepath, sheet_name="Summary", header=None)
    except FileNotFoundError:
        print(f"错误: 未找到数据文件 '{filepath}'。")
        return pd.DataFrame()
    parsed_data = []
    current_cosine, current_top_k = None, None
    for index, row in df_raw.iterrows():
        row_str = str(row.iloc[0])
        if "实验组:" in row_str:
            match = re.search(r"cs(\d+\.?\d*).*tk(\d+)", row_str)
            if match:
                current_cosine, current_top_k = (
                    float(match.group(1)),
                    int(match.group(2)),
                )
            continue
        if "平均值 (Mean Win Rate)" in row_str:
            data_start_index = index + 2
            for data_idx in range(data_start_index, len(df_raw)):
                data_row = df_raw.iloc[data_idx]
                system_name = data_row.iloc[0]
                if pd.isna(system_name) or "实验组:" in str(system_name):
                    break
                try:
                    values = [float(str(v).strip("%")) for v in data_row.iloc[1:5]]
                    parsed_data.append(
                        [system_name, current_cosine, current_top_k] + values
                    )
                except (ValueError, TypeError):
                    continue
    if not parsed_data:
        print("警告: 从Excel文件中未能解析出任何有效数据。")
        return pd.DataFrame()
    return pd.DataFrame(
        parsed_data,
        columns=[
            "system",
            "cosine",
            "top_k",
            "Comprehensiveness",
            "Diversity",
            "Empowerment",
            "Overall",
        ],
    )


def get_dominance_data(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    best_system_idx = df.loc[df.groupby(["cosine", "top_k"])[metric].idxmax()]
    dominance_pivot = best_system_idx.pivot(
        index="top_k", columns="cosine", values="system"
    )
    return dominance_pivot


def construct_prompt(query: str, answer1: str, answer2: str) -> str:
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


async def call_deepseek(prompt: str, client: OpenAI) -> dict:
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
            return {"content": ""}

    return await asyncio.to_thread(sync_call)


def parse_evaluation_result(response_text: str) -> dict:
    if not response_text:
        return {}
    try:
        return json.loads(response_text)
    except Exception:
        try:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start != -1 and json_end != -1:
                return json.loads(response_text[json_start:json_end])
            return {}
        except Exception:
            return {}


# --- 3. 核心逻辑 ---
def extract_champion_answers(
    query: str, dominance_pivot: pd.DataFrame, f_names_map: dict, results_root_dir: Path
) -> list:
    champions, file_cache = [], {}
    for cosine in dominance_pivot.columns:
        for top_k in dominance_pivot.index:
            system = dominance_pivot.loc[top_k, cosine]
            if pd.isna(system):
                continue
            f_name = f_names_map.get(system)
            if not f_name:
                continue
            result_dir = results_root_dir / f"cs{cosine}_tk{top_k}"
            json_path = result_dir / f"{f_name}_result.json"
            answer_text = "Answer Not Found in JSON File"
            if str(json_path) not in file_cache:
                if json_path.exists():
                    try:
                        with open(json_path, "r", encoding="utf-8") as f:
                            file_cache[str(json_path)] = json.load(f)
                    except (json.JSONDecodeError, IOError):
                        file_cache[str(json_path)] = None
                else:
                    file_cache[str(json_path)] = None
            if file_cache.get(str(json_path)):
                for record in file_cache[str(json_path)]:
                    if record.get("query", "").strip() == query.strip():
                        answer_text = record.get(
                            "result", "Answer text missing in record."
                        )
                        break
            champions.append(
                {
                    "id": f"{system} (cs={cosine}, tk={top_k})",
                    "system": system,
                    "cosine": cosine,
                    "top_k": top_k,
                    "answer_text": answer_text,
                }
            )
    return champions


async def run_tournament_for_query(query: str, champions: list, client: OpenAI) -> dict:
    if len(champions) < 2:
        return {}
    scoreboard = {champ["id"]: 0 for champ in champions}
    matchups = list(itertools.combinations(champions, 2))
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    async def run_match(champ1, champ2):
        async with semaphore:
            c1, c2, slot1, slot2 = (
                (champ1, champ2, "Answer 1", "Answer 2")
                if random.choice([True, False])
                else (champ2, champ1, "Answer 2", "Answer 1")
            )
            prompt = construct_prompt(query, c1["answer_text"], c2["answer_text"])
            api_response = await call_deepseek(prompt, client)
            eval_result = parse_evaluation_result(api_response.get("content", ""))
            winner_slot = eval_result.get("Overall", {}).get("Winner")
            if winner_slot == slot1:
                return c1["id"]
            elif winner_slot == slot2:
                return c2["id"]
            return None

    tasks = [run_match(m[0], m[1]) for m in matchups]
    winners = await tqdm_asyncio.gather(
        *tasks, desc=f"  - 循环赛对决中 ({len(tasks)}场)", leave=False
    )
    for winner_id in winners:
        if winner_id in scoreboard:
            scoreboard[winner_id] += 1
    return scoreboard


# --- 4. 主执行函数 ---
async def main():
    """主执行函数，直接从顶部配置区读取路径。"""
    load_dotenv(dotenv_path=ENV_FILE_PATH)
    if not os.getenv("ALIYUN_API_KEY"):
        print(
            f"错误: 找不到 ALIYUN_API_KEY。请检查 .env 文件 (路径: {ENV_FILE_PATH}) 或环境变量设置。"
        )
        return

    df_full = load_data_from_excel(EXCEL_FILE_PATH)
    if df_full.empty:
        return

    output_dir = Path(OUTPUT_DIRECTORY)
    output_dir.mkdir(exist_ok=True, parents=True)
    results_root_dir = Path(RESULTS_ROOT_DIR)

    f_names_map = {
        "StruMech": "stru_mech",
        "_Manual": "_manual",
        "_Split": "_split",
        "LightRAG": "lightrag",
        "Only_Manual": "only_manual",
    }

    # --- 修正：从一个原始JSON文件中获取问题列表 ---
    try:
        # 智能地寻找一个存在的JSON文件来读取问题列表
        sample_json_path = next(results_root_dir.rglob("*_result.json"))
        with open(sample_json_path, "r", encoding="utf-8") as f:
            sample_data = json.load(f)
        unique_queries = sorted(
            [item["query"] for item in sample_data if "query" in item]
        )
        if not unique_queries:
            raise FileNotFoundError
        print(
            f"成功从 {sample_json_path.name} 文件中加载了 {len(unique_queries)} 个唯一问题。"
        )
    except (StopIteration, FileNotFoundError, json.JSONDecodeError):
        print(
            f"错误: 无法在 '{results_root_dir}' 目录下找到任何有效的 '*_result.json' 文件来读取问题列表。"
        )
        return
    # --- 修正结束 ---

    dominance_pivot = get_dominance_data(df_full, TARGET_METRIC)

    final_rankings = []
    client = OpenAI(
        api_key=os.getenv("ALIYUN_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    for query in tqdm(unique_queries, desc="处理问题总进度"):
        tqdm.write(f"\n{'=' * 20} 正在为问题 '{query[:40]}...' 举办冠军赛 {'=' * 20}")
        champions = extract_champion_answers(
            query, dominance_pivot, f_names_map, results_root_dir
        )
        if len(champions) < 2:
            tqdm.write(
                f"警告: 问题 '{query[:40]}...' 找到的冠军答案少于2个，无法比赛。"
            )
            continue
        scoreboard = await run_tournament_for_query(query, champions, client)
        if scoreboard:
            sorted_champs = sorted(
                scoreboard.items(), key=lambda item: item[1], reverse=True
            )
            for rank, (champ_id, wins) in enumerate(sorted_champs):
                match = re.search(r"(.+?)\s\(cs=([\d.]+),\s*tk=(\d+)\)", champ_id)
                system, cosine, top_k = (
                    (match.group(1), float(match.group(2)), int(match.group(3)))
                    if match
                    else ("Unknown", 0, 0)
                )
                final_rankings.append(
                    {
                        "query": query,
                        "rank": rank + 1,
                        "system": system,
                        "cosine": cosine,
                        "top_k": top_k,
                        "win_count": wins,
                        "total_matches": len(champions) - 1,
                    }
                )

    if final_rankings:
        report_df = pd.DataFrame(final_rankings)
        output_path = output_dir / "Championship_Rankings_Final.csv"
        report_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\n{'🎉'*10} 所有冠军赛评估已全部完成！ {'🎉'*10}")
        print(f"最终排名报告已保存至: {output_path}")
    else:
        print("\n评估未能生成任何结果。")

# --- 5. 脚本入口 ---
if __name__ == "__main__":
    asyncio.run(main())