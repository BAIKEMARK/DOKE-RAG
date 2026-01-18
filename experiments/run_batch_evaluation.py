# -*- coding: utf-8 -*-

"""
全自动批量实验运行脚本
========================

功能:
1. 对预定义的参数网格 (cosine_threshold, top_k) 进行遍历。
2. 对每一组网格参数，运行一组核心配置 (Core Configurations)。
3. 通过命令行参数 `--group` 选择要运行的核心配置组 ('group1' 或 'group2')。
4. 自动创建结构化的输出目录来保存每次实验的结果。
5. 串行执行，确保每次实验完全独立，避免数据混淆。
6. 智能断点续跑：自动跳过已有有效结果的实验，并重新运行失败的或结果为空的实验。
7. 路径自适应：自动寻找与脚本在同一目录下的问题文件。

如何使用:
1. 将此文件保存为 `run_batch_experiment.py`。
2. 确保你的环境中已安装 `lightrag` 及其依赖。
3. 将问题文件 (例如 `questions_updated.json`) 与此脚本放在同一个文件夹下。
4. 设置好运行 `group1` 所需的环境变量。
5. 打开终端，运行命令:
   python experiments/run_batch_evaluation.py --group group1 --base_output_dir "./evaluation/results/batch_experiment"

6. 设置好运行 `group2` 所需的环境变量。
7. 在终端中，运行命令:
   python experiments/run_batch_evaluation.py --group group2 --base_output_dir "./evaluation/results/batch_experiment"
"""

import os
import json
import argparse
import itertools
import traceback
from pathlib import Path
from doke_rag.core import LightRAG, QueryParam
from doke_rag.core.llm.openai import openai_complete_if_cache
from doke_rag.core.llm.ollama import ollama_embed
from doke_rag.core.utils import EmbeddingFunc, always_get_an_event_loop, TokenTracker
from doke_rag.config.paths import WORKING_DIR, RESULTS_DIR, ensure_dir

# ==============================================================================
# 1. 在这里定义你的所有实验参数
# ==============================================================================

# 参数网格
GRID_PARAMS = {
    "cosine_threshold": [0.2, 0.4, 0.6, 0.8],
    "top_k": [20, 40, 60, 80],
}

# 核心配置组: 分成两组，通过命令行参数选择
# 注意：working_dir 应使用相对于项目根目录的路径
# 可以通过环境变量 WORKING_DIR 设置基础目录，或直接使用相对路径
CONFIG_GROUPS = {
    "group1": [
        # 示例配置：取消注释并根据需要修改路径
        # {
        #     "mode": "hybrid",
        #     "working_dir": "./data/run_data/experiment1/merged_no_textbook",
        #     "f_name": "stru_mech",
        # },
        # {
        #     "mode": "hybrid",
        #     "working_dir": "./data/run_data/experiment1/unmerged_no_textbook",
        #     "f_name": "_manual",
        # },
        {
            "mode": "hybrid",
            "working_dir": "./data/run_data/experiment1/unsplited_no_textbook",
            "f_name": "_split",
        },
    ],
    "group2": [
        {
            "mode": "hybrid",
            "working_dir": "./data/run_data/experiment2/lightrag_baseline",
            "f_name": "lightrag",
        },
        {
            "mode": "naive",
            "working_dir": "./data/run_data/experiment2/lightrag_baseline",
            "f_name": "naive",
        },
        {
            "mode": "hybrid",
            "working_dir": "./data/run_data/experiment2/manual_only",
            "f_name": "only_manual",
        },
    ],
}

# ==============================================================================
# 2. 辅助函数和全局对象
# ==============================================================================

token_tracker = TokenTracker()


async def llm_model_func_aliyun(prompt, **kwargs) -> str:
    # 确保 API Key 从环境变量中正确读取
    api_key = os.getenv("ALIYUN_API_KEY")
    if not api_key:
        raise ValueError("ALIYUN_API_KEY environment variable not set.")
    return await openai_complete_if_cache(
        "deepseek-r1-distill-qwen-14b",
        prompt,
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        token_tracker=token_tracker,
        **kwargs,
    )


def extract_queries_from_json(file_path: str) -> list[str]:
    """从 JSON 文件中安全地读取问题列表。"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("questions", [])
    except FileNotFoundError:
        print(f"Error: Questions file not found at {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return []


async def process_query(
    query_text: str, rag_instance: LightRAG, query_param: QueryParam
) -> tuple[dict | None, dict | None]:
    """处理单个查询，返回结果和错误信息。"""
    token_tracker.reset()
    try:
        result = await rag_instance.aquery(query_text, param=query_param)
        usage = token_tracker.get_usage()
        return {
            "query": query_text,
            "result": result,
            "input_tokens": usage["prompt_tokens"],
            "output_tokens": usage["completion_tokens"],
        }, None
    except Exception as e:
        return None, {"query": query_text, "error": str(e)}


def run_queries_and_save_to_json(
    queries: list[str],
    rag_instance: LightRAG,
    query_param: QueryParam,
    output_file: str,
    error_file: str,
):
    """在一个事件循环中运行所有查询并保存结果。"""
    loop = always_get_an_event_loop()
    with (
        open(output_file, "w", encoding="utf-8") as result_file,
        open(error_file, "w", encoding="utf-8") as err_file,
    ):
        result_file.write("[\n")
        first_entry = True
        error_entries = []

        for i, query_text in enumerate(queries):
            print(f"    - Processing query {i + 1}/{len(queries)}...")
            result, error = loop.run_until_complete(
                process_query(query_text, rag_instance, query_param)
            )
            if result:
                if not first_entry:
                    result_file.write(",\n")
                json.dump(result, result_file, ensure_ascii=False, indent=4)
                first_entry = False
            elif error:
                error_entries.append(error)

        if error_entries:
            json.dump(error_entries, err_file, ensure_ascii=False, indent=4)

        result_file.write("\n]")


# ==============================================================================
# 3. 单次实验的核心逻辑
# ==============================================================================


def run_single_experiment(config: dict):
    """
    运行单次实验的核心函数。
    'config' 字典包含了运行一次所需的所有参数。
    """
    print(
        f"--- Initializing experiment: {config['f_name']} (cs={config['cosine_threshold']}, tk={config['top_k']}) ---"
    )
    print(f"    Working directory: {config['working_dir']}")

    # 1. 初始化 LightRAG 实例
    rag = LightRAG(
        working_dir=config["working_dir"],
        llm_model_func=llm_model_func_aliyun,
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts,
                embed_model="nomic-embed-text:latest",
                host="http://localhost:11434",
            ),
        ),
        vector_db_storage_cls_kwargs={
            "cosine_better_than_threshold": config["cosine_threshold"]
        },
        llm_model_max_async=4,
        llm_model_max_token_size=32768,
        entity_extract_max_gleaning=1,
        entity_summary_to_max_tokens=500,
        addon_params={"example_number": 3, "insert_batch_size": 50},
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
        enable_llm_cache=False,
    )

    # 2. 设置查询参数
    query_param = QueryParam(
        mode=config["mode"],
        top_k=config["top_k"],
        max_token_for_text_unit=4000,
        max_token_for_global_context=4000,
        max_token_for_local_context=4000,
    )

    # 3. 准备输出文件路径
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, f"{config['f_name']}_result.json")
    error_file_path = os.path.join(output_dir, f"{config['f_name']}_errors.json")

    # 4. 执行查询并保存结果
    print(f"    Starting queries...")
    run_queries_and_save_to_json(
        config["queries"], rag, query_param, output_file_path, error_file_path
    )
    print(f"--- ✅ Finished experiment. Results saved in: {output_dir} ---")


# ==============================================================================
# 4. 批量实验运行器
# ==============================================================================

def is_result_file_valid(filepath: str) -> bool:
    """
    检查结果文件是否存在，并且是否包含有效的、非空的内容。
    """
    # 1. 检查文件是否存在且大小大于一个很小的值 (空的 "[]" 大约是2-4字节)
    if not os.path.exists(filepath) or os.path.getsize(filepath) < 5:
        return False

    # 2. 读取并尝试解析JSON，确保它是一个非空列表
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 必须是列表且长度大于0
        if isinstance(data, list) and len(data) > 0:
            return True
    except (json.JSONDecodeError, IOError):
        # 如果文件损坏或无法读取，视为无效
        return False

    return False

def main(args):
    """批量实验运行器的入口函数。"""
    # 如果 questions_file 是相对路径，则将其解析为相对于脚本所在目录的路径
    if not os.path.isabs(args.questions_file):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.questions_file = os.path.join(script_dir, args.questions_file)

    # 根据命令行参数选择要运行的配置组
    selected_group = CONFIG_GROUPS.get(args.group)
    if not selected_group:
        print(
            f"Error: Invalid group '{args.group}'. Please choose from {list(CONFIG_GROUPS.keys())}"
        )
        return

    # 生成所有 grid 参数组合
    keys, values = zip(*GRID_PARAMS.items())
    grid_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    total_runs = len(grid_combinations) * len(selected_group)
    run_counter = 0

    # 一次性读取问题文件，供所有实验使用
    queries = extract_queries_from_json(args.questions_file)
    if not queries:
        print("No queries found. Exiting.")
        return

    print(f"Loaded {len(queries)} questions from '{args.questions_file}'")
    print(
        f"Starting batch for '{args.group}' with a total of {total_runs} experiments..."
    )

    # 外层循环：遍历参数网格 (e.g., cs=0.2, tk=20)
    for grid_combo in grid_combinations:
        cs = grid_combo["cosine_threshold"]
        tk = grid_combo["top_k"]

        # 为当前网格组合创建独立的输出文件夹
        current_output_dir = os.path.join(args.base_output_dir, f"cs{cs}_tk{tk}")

        # 内层循环：遍历选择的核心配置组 (e.g., stru_mech, _manual)
        for core_config in selected_group:
            run_counter += 1
            print(
                f"\n====================== [ Checking {run_counter} / {total_runs} ] ======================"
            )

            # 使用更智能的断点续跑检查
            expected_output_file = os.path.join(current_output_dir, f"{core_config['f_name']}_result.json")

            if is_result_file_valid(expected_output_file):
                print(f"✅ Skipping: Valid result for '{core_config['f_name']}' (cs={cs}, tk={tk}) already exists.")
                continue
            else:
                 print(f"🏃‍♂️ Running: Result for '{core_config['f_name']}' (cs={cs}, tk={tk}) is missing or invalid.")

            # 组合成一个完整的配置字典
            full_config = {
                **grid_combo,
                **core_config,
                "output_dir": current_output_dir,
                "queries": queries,
            }

            try:
                # 运行单次实验
                run_single_experiment(full_config)
            except Exception:
                print(f"!!!!!!!!!!!!!! FATAL ERROR IN EXPERIMENT !!!!!!!!!!!!!!")
                print(f"Config that failed: {core_config['f_name']}, cs={cs}, tk={tk}")
                # 打印详细的错误堆栈信息
                traceback.print_exc()
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("Skipping to the next experiment...")
                continue

    print(
        f"\n🎉 All {total_runs} experiments for group '{args.group}' have been completed."
    )


# ==============================================================================
# 5. 脚本入口
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a batch of LightRAG experiments from a predefined grid search.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--group",
        type=str,
        required=True,
        choices=["group1", "group2"],
        help="Specify which configuration group to run ('group1' or 'group2').",
    )

    parser.add_argument(
        "--questions_file",
        type=str,
        default="questions_updated.json",
        help="Path to the JSON file containing questions. \nDefaults to a file with this name in the script's directory.",
    )

    parser.add_argument('--base_output_dir', type=str, required=True,
                        help="The base directory where all result folders will be created.\n"
                             "Example: D:\\RAG_Results")

    args = parser.parse_args()
    main(args)