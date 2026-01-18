import os
import json
from pathlib import Path
from doke_rag.core import LightRAG, QueryParam
from doke_rag.core.llm.openai import openai_complete_if_cache
from doke_rag.core.utils import EmbeddingFunc, always_get_an_event_loop, TokenTracker
from doke_rag.core.llm.ollama import ollama_embed
from doke_rag.config.paths import WORKING_DIR, ensure_dir

token_tracker = TokenTracker()

async def llm_model_func_aliyun(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        # "deepseek-r1",
        # "qwen3-235b-a22b-thinking-2507",
        "deepseek-r1-distill-qwen-14b",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("ALIYUN_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里云
        token_tracker=token_tracker,
        **kwargs,
    )

async def llm_model_func_deepseek(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        "deepseek-reasoner",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
        token_tracker=token_tracker,
        temperature=1.0,

        **kwargs,
    )

def extract_queries_from_json(file_path):
    """
    从 JSON 文件中读取问题列表，JSON 格式如下：
    {
      "questions": [ "问题1", "问题2", ... ]
    }
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    queries = data.get("questions", [])
    return queries

async def process_query(query_text, rag_instance, query_param):
    token_tracker.reset()  # 在每次处理前清零，保证只统计这一个问题
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
    queries, rag_instance, query_param, output_file, error_file
):
    loop = always_get_an_event_loop()
    with open(output_file, "w", encoding="utf-8") as result_file, open(
        error_file, "w", encoding="utf-8"
    ) as err_file:
        result_file.write("[\n")
        first_entry = True

        for query_text in queries:
            result, error = loop.run_until_complete(
                process_query(query_text, rag_instance, query_param)
            )
            if result:
                if not first_entry:
                    result_file.write(",\n")
                json.dump(result, result_file, ensure_ascii=False, indent=4)
                first_entry = False
            elif error:
                json.dump(error, err_file, ensure_ascii=False, indent=4)
                err_file.write("\n")

        result_file.write("\n]")

if __name__ == "__main__":
    # 配置查询模式和工作目录后缀
    # 通过环境变量 WORKING_DIR 设置工作目录
    # 示例：在 .env 中设置 WORKING_DIR=./data/run_data/experiment1
    mode = "hybrid"
    f_name = "_split"  # 用于标识不同的实验配置

    ensure_dir(WORKING_DIR)

    # 初始化 LightRAG 实例，并传入 llm_model_func 与 embedding_func
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func_aliyun,
        llm_model_max_async=4,
        llm_model_max_token_size=32768,
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts,
                embed_model="nomic-embed-text:latest",
                host="http://localhost:11434",
            ),
        ),
        entity_extract_max_gleaning=1,
        entity_summary_to_max_tokens=500,
        addon_params={
            "example_number": 3,
            "insert_batch_size": 50,
            # "entity_types": ["method", "concept", "equation", "structure", "constraint", "process", "step"],
            # "language": "Simplified Chinese",
        },
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
        vector_db_storage_cls_kwargs={"cosine_better_than_threshold": 0.2},
        enable_llm_cache=False,
    )

    token_tracker.reset()
    query_param = QueryParam(
        mode=mode,
        top_k=40,
        max_token_for_text_unit=4000,
        max_token_for_global_context=4000,
        max_token_for_local_context=4000,
    )

    base_dir = "QA"
    # json_file = f"questions_en.json"
    json_file = f"questions_updated.json"
    queries = extract_queries_from_json(json_file)
    print("Total number of queries:", len(queries))
    run_queries_and_save_to_json(
        queries, rag, query_param, f"ds_14b_split_cs20_tk40/{f_name}_result.json", f"ds_14b_split_cs20_tk40/{f_name}_errors.json"
        # queries, rag, query_param, f"{base_dir}/naive_{f_name}_result.json", f"{base_dir}/{f_name}_errors.json"
    )
    print("Overall Token usage:", token_tracker.get_usage())