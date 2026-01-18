# import asyncio
# import os
# import json
# from typing import List, Union
# from pydantic import BaseModel, ValidationError
# from openai import AsyncOpenAI, APIError
# from dotenv import load_dotenv
#
# # --- 1. 加载环境变量 ---
# load_dotenv()
#
# # --- 2. 配置模型和 API ---
# # 阿里云百炼 OpenAI 兼容模式的配置
# # 注意：你需要根据实际情况替换 model_name
# # 可用模型如: qwen-turbo, qwen-plus, qwen-max, qwen-max-longcontext 等
# BAILIAN_API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# BAILIAN_MODEL_NAME = "qwen-plus"
# BAILIAN_API_KEY = os.getenv("DASHSCOPE_API_KEY")
#
# # --- 3. Pydantic 模型和提示词 (与您提供的版本基本一致) ---
#
# class EntityTypesResponse(BaseModel):
#     """实体类型响应模型，用于验证 JSON 输出"""
#     entity_types: List[str]
#
# # 为提高AI理解一致性，将JSON提示词末尾的 "JSON response format:" 改为 "JSON RESPONSE:"
ENTITY_TYPE_GENERATION_PROMPT = """
The goal is to study the connections and relations between the entity types and their features in order to understand all available information from the text.
The user's task is to {task}.
As part of the analysis, you want to identify the entity types present in the following text.
The entity types must be relevant to the user task.
Avoid general entity types such as "other" or "unknown".
This is VERY IMPORTANT: Do not generate redundant or overlapping entity types. For example, if the text contains "company" and "organization" entity types, you should return only one of them.
Don't worry about quantity, always choose quality over quantity. And make sure EVERYTHING in your answer is relevant to the context of entity extraction.
And remember, it is ENTITY TYPES what we need.
Return the entity types in as a list of comma separated strings.

EXAMPLE 1
Task: Determine the connections and organizational hierarchy within the specified community.
Text: Example_Org_A is a company in Sweden. Example_Org_A's director is Example_Individual_B.
RESPONSE:
organization, person

EXAMPLE 2
Task: Identify the key concepts, principles, and arguments shared among different philosophical schools of thought.
Text: Rationalism, epitomized by thinkers such as René Descartes, holds that reason is the primary source of knowledge.
RESPONSE:
concept, person, school of thought

EXAMPLE 3
Task: Identify the full range of basic forces, factors, and trends that would indirectly shape an issue.
Text: Industry leaders such as Panasonic are vying for supremacy in the battery production sector.
RESPONSE:
organization, technology, sectors, investment strategies

REAL DATA: The following section is the real data. You should use only this real data to prepare your answer. Generate Entity Types only.
Task: {task}
Text: {input_text}
RESPONSE:
"""

ENTITY_TYPE_GENERATION_JSON_PROMPT = """
The goal is to study the connections and relations between the entity types and their features in order to understand all available information from the text.
The user's task is to {task}.
As part of the analysis, you want to identify the entity types present in the following text.
The entity types must be relevant to the user task.
Avoid general entity types such as "other" or "unknown".
This is VERY IMPORTANT: Do not generate redundant or overlapping entity types. For example, if the text contains "company" and "organization" entity types, you should return only one of them.
Don't worry about quantity, always choose quality over quantity. And make sure EVERYTHING in your answer is relevant to the context of entity extraction.
Return the entity types in JSON format with "entity_types" as the key and the entity types as an array of strings.

EXAMPLE 1
Task: Determine the connections and organizational hierarchy within the specified community.
Text: Example_Org_A is a company in Sweden. Example_Org_A's director is Example_Individual_B.
JSON RESPONSE:
{{"entity_types": ["organization", "person"]}}

EXAMPLE 2
Task: Identify the key concepts, principles, and arguments shared among different philosophical schools of thought.
Text: Rationalism, epitomized by thinkers such as René Descartes, holds that reason is the primary source of knowledge.
JSON RESPONSE:
{{"entity_types": ["concept", "person", "school of thought"]}}

EXAMPLE 3
Task: Identify the full range of basic forces, factors, and trends that would indirectly shape an issue.
Text: Industry leaders such as Panasonic are vying for supremacy in the battery production sector.
JSON RESPONSE:
{{"entity_types": ["organization", "technology", "sectors", "investment strategies"]}}

REAL DATA: The following section is the real data. You should use only this real data to prepare your answer. Generate Entity Types only.
Task: {task}
Text: {input_text}
JSON RESPONSE:
"""
#
# DEFAULT_TASK = "Identify the relations and structure of the community of interest, specifically within the {domain} domain."
#
# # --- 4. 重构后的核心函数 ---
#
# async def generate_entity_types(
#         client: AsyncOpenAI,
#         model_name: str,
#         domain: str,
#         persona: str,
#         docs: Union[str, List[str]],
#         task: str = DEFAULT_TASK,
#         json_mode: bool = False,
# ) -> List[str]:
#     """
#     使用 LLM API 从给定的文档集合中生成实体类型。
#     返回一个字符串列表，如果失败则返回空列表。
#     """
#     if not BAILIAN_API_KEY:
#         print("错误: 环境变量 DASHSCOPE_API_KEY 未设置。")
#         return []
#
#     # 准备输入
#     formatted_task = task.format(domain=domain)
#     docs_str = "\n".join(docs) if isinstance(docs, list) else docs
#
#     prompt = (
#         ENTITY_TYPE_GENERATION_JSON_PROMPT if json_mode else ENTITY_TYPE_GENERATION_PROMPT
#     ).format(task=formatted_task, input_text=docs_str)
#
#     messages = [
#         {"role": "system", "content": persona},
#         {"role": "user", "content": prompt},
#     ]
#
#     try:
#         # 调用 LLM API
#         if json_mode:
#             # 请求 JSON 输出
#             response = await client.chat.completions.create(
#                 model=model_name,
#                 messages=messages,
#                 response_format={"type": "json_object"},
#             )
#             content = response.choices[0].message.content
#             # 解析并验证 JSON
#             try:
#                 parsed_model = EntityTypesResponse.model_validate_json(content)
#                 return parsed_model.entity_types
#             except (ValidationError, json.JSONDecodeError) as e:
#                 print(f"错误: JSON 解析或验证失败。错误: {e}\n原始输出: {content}")
#                 return []
#         else:
#             # 请求文本输出
#             response = await client.chat.completions.create(
#                 model=model_name,
#                 messages=messages,
#             )
#             content = response.choices[0].message.content
#             # 将逗号分隔的字符串转换为列表
#             if content:
#                 return [item.strip() for item in content.split(',')]
#             return []
#
#     except APIError as e:
#         print(f"错误: 调用 API 失败。状态码: {e.status_code}, 响应: {e.response}")
#         return []
#     except Exception as e:
#         print(f"发生未知错误: {e}")
#         return []
#
#
# # --- 5. 更新后的使用示例 ---
#
# async def main():
#     """主执行函数"""
#     # 初始化 OpenAI 客户端以连接到百炼 API
#     client = AsyncOpenAI(api_key=BAILIAN_API_KEY, base_url=BAILIAN_API_BASE_URL)
#
#     # 设置参数
#     domain = "technology and business"
#     persona = "You are an expert analyst skilled in identifying and categorizing entities from text documents."
#     docs = [
#         "Apple Inc. is a technology company founded by Steve Jobs. The company is headquartered in Cupertino, California.",
#         "Microsoft Corporation develops software products. Bill Gates was one of its founders. The company is based in Redmond, Washington.",
#         "Google LLC operates search engines and cloud services. It was founded by Larry Page and Sergey Brin at Stanford University."
#     ]
#
#     # --- 生成实体类型（文本模式）---
#     print("=== 文本模式 ===")
#     entity_types_text = await generate_entity_types(
#         client=client,
#         model_name=BAILIAN_MODEL_NAME,
#         domain=domain,
#         persona=persona,
#         docs=docs,
#         json_mode=False
#     )
#     print(f"抽取的实体类型: {entity_types_text}")
#     print("-" * 20)
#
#     # # --- 生成实体类型（JSON 模式）---
#     # print("\n=== JSON 模式 ===")
#     # entity_types_json = await generate_entity_types(
#     #     client=client,
#     #     model_name=BAILIAN_MODEL_NAME,
#     #     domain=domain,
#     #     persona=persona,
#     #     docs=docs,
#     #     json_mode=True
#     # )
#     # print(f"抽取的实体类型: {entity_types_json}")
#     # print("-" * 20)
#
#
# if __name__ == "__main__":
#     asyncio.run(main())

import asyncio
import os
import json
from typing import List, Dict, Any, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError
from openai import AsyncOpenAI, APIError
from dotenv import load_dotenv
import ollama # 引入 ollama 库

# --- 1. 全局配置 ---
load_dotenv()

# Ollama 配置
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text" # 替换为你已下载的 Ollama 嵌入模型

# 阿里云百炼 LLM 配置
BAILIAN_API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
BAILIAN_MODEL_NAME = "qwen-max"
BAILIAN_API_KEY = os.getenv("ALIYUN_API_KEY")


# --- 2. Ollama 嵌入模型实现 ---

class OllamaEmbeddingModel:
    """使用本地 Ollama 服务的嵌入模型"""

    def __init__(self, model_name: str, host: str = OLLAMA_HOST):
        self.model_name = model_name
        try:
            self.client = ollama.AsyncClient(host=host)
            print(f"Ollama 客户端已连接至 {host}")
        except Exception as e:
            print(f"无法连接到 Ollama 服务在 {host}。请确保 Ollama 正在运行。错误: {e}")
            self.client = None

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文本"""
        if not self.client:
            raise ConnectionError("Ollama 客户端未初始化。")

        embeddings = []
        # 使用 asyncio.gather 并发处理嵌入请求
        tasks = [self.client.embeddings(model=self.model_name, prompt=text) for text in texts]

        try:
            results = await asyncio.gather(*tasks)
            for res in results:
                embeddings.append(res['embedding'])
            return embeddings
        except Exception as e:
            print(f"从 Ollama 获取嵌入向量时出错: {e}")
            # 如果部分失败，也可以选择返回成功的部分，这里为简单起见直接抛出异常
            raise


# --- 3. 文档分块与选择 (新增部分) ---

def _sample_chunks_from_embeddings(
        text_chunks: pd.DataFrame,
        embeddings: np.ndarray,
        k: int = 15,
) -> pd.DataFrame:
    """从嵌入向量中采样文本块，选择离中心点最近的k个"""
    if embeddings.shape[0] < k:
        print(f"警告: 嵌入向量数量 ({embeddings.shape[0]}) 小于 k ({k})，将返回所有可用块。")
        return text_chunks

    center = np.mean(embeddings, axis=0)
    distances = np.linalg.norm(embeddings - center, axis=1)

    # argsort 返回的是排序后的索引
    nearest_indices = np.argsort(distances)[:k]

    # 使用 .iloc 从原始 DataFrame 中选择行
    return text_chunks.iloc[nearest_indices]


def create_text_chunks(
        documents: List[str],
        chunk_size: int = 200,
        overlap: int = 50
) -> pd.DataFrame:
    """将文档分割成文本块"""
    chunks = []
    chunk_id = 0
    for doc_id, document in enumerate(documents):
        words = document.split()
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = ' '.join(words[start:end])
            chunks.append({'id': chunk_id, 'text': chunk_text})
            chunk_id += 1
            start += chunk_size - overlap
            if end >= len(words):
                break
    return pd.DataFrame(chunks)


async def load_docs_with_auto_selection(
        documents: List[str],
        embedding_model: OllamaEmbeddingModel,
        k: int = 15,
        chunk_size: int = 200,
        overlap: int = 50,
) -> List[str]:
    """使用嵌入方法加载和选择文档块"""
    # 1. 将文档分块
    print("正在分块文档...")
    chunks_df = create_text_chunks(documents, chunk_size, overlap)
    print(f"生成了 {len(chunks_df)} 个文本块")

    if chunks_df.empty:
        print("警告: 未生成任何文本块。")
        return []

    # 2. 计算嵌入向量 (对所有块)
    print(f"正在使用 Ollama 模型 '{embedding_model.model_name}' 计算嵌入向量...")
    text_list_for_embedding = chunks_df["text"].tolist()
    embeddings_list = await embedding_model.embed_batch(text_list_for_embedding)
    embeddings = np.array(embeddings_list)
    print(f"嵌入向量形状: {embeddings.shape}")

    # 3. 基于嵌入选择最具代表性的文档块
    print(f"基于嵌入选择最具代表性的 {k} 个文档块...")
    selected_chunks_df = _sample_chunks_from_embeddings(chunks_df, embeddings, k=k)

    # 4. 转换为最终格式
    selected_texts = [
        text.replace("{", "{{").replace("}", "}}")
        for text in selected_chunks_df["text"]
    ]
    print(f"最终选择了 {len(selected_texts)} 个文档块")
    return selected_texts


# --- 4. 实体类型生成  ---
class EntityTypesResponse(BaseModel):
    entity_types: List[str]
ENTITY_TYPE_GENERATION_PROMPT = """
The goal is to study the connections and relations between the entity types and their features in order to understand all available information from the text.
The user's task is to {task}.
As part of the analysis, you want to identify the entity types present in the following text.
The entity types must be relevant to the user task.
Avoid general entity types such as "other" or "unknown".
This is VERY IMPORTANT: Do not generate redundant or overlapping entity types. For example, if the text contains "company" and "organization" entity types, you should return only one of them.
Don't worry about quantity, always choose quality over quantity. And make sure EVERYTHING in your answer is relevant to the context of entity extraction.
And remember, it is ENTITY TYPES what we need.
Return the entity types in as a list of comma separated strings.

EXAMPLE 1
Task: Determine the connections and organizational hierarchy within the specified community.
Text: Example_Org_A is a company in Sweden. Example_Org_A's director is Example_Individual_B.
RESPONSE:
organization, person

EXAMPLE 2
Task: Identify the key concepts, principles, and arguments shared among different philosophical schools of thought.
Text: Rationalism, epitomized by thinkers such as René Descartes, holds that reason is the primary source of knowledge.
RESPONSE:
concept, person, school of thought

EXAMPLE 3
Task: Identify the full range of basic forces, factors, and trends that would indirectly shape an issue.
Text: Industry leaders such as Panasonic are vying for supremacy in the battery production sector.
RESPONSE:
organization, technology, sectors, investment strategies

REAL DATA: The following section is the real data. You should use only this real data to prepare your answer. Generate Entity Types only.
Task: {task}
Text: {input_text}
RESPONSE:
"""

ENTITY_TYPE_GENERATION_JSON_PROMPT = """
The goal is to study the connections and relations between the entity types and their features in order to understand all available information from the text.
The user's task is to {task}.
As part of the analysis, you want to identify the entity types present in the following text.
The entity types must be relevant to the user task.
Avoid general entity types such as "other" or "unknown".
This is VERY IMPORTANT: Do not generate redundant or overlapping entity types. For example, if the text contains "company" and "organization" entity types, you should return only one of them.
Don't worry about quantity, always choose quality over quantity. And make sure EVERYTHING in your answer is relevant to the context of entity extraction.
Return the entity types in JSON format with "entity_types" as the key and the entity types as an array of strings.

EXAMPLE 1
Task: Determine the connections and organizational hierarchy within the specified community.
Text: Example_Org_A is a company in Sweden. Example_Org_A's director is Example_Individual_B.
JSON RESPONSE:
{{"entity_types": ["organization", "person"]}}

EXAMPLE 2
Task: Identify the key concepts, principles, and arguments shared among different philosophical schools of thought.
Text: Rationalism, epitomized by thinkers such as René Descartes, holds that reason is the primary source of knowledge.
JSON RESPONSE:
{{"entity_types": ["concept", "person", "school of thought"]}}

EXAMPLE 3
Task: Identify the full range of basic forces, factors, and trends that would indirectly shape an issue.
Text: Industry leaders such as Panasonic are vying for supremacy in the battery production sector.
JSON RESPONSE:
{{"entity_types": ["organization", "technology", "sectors", "investment strategies"]}}

REAL DATA: The following section is the real data. You should use only this real data to prepare your answer. Generate Entity Types only.
Task: {task}
Text: {input_text}
JSON RESPONSE:
"""

DEFAULT_TASK = "Identify the relations and structure of the community of interest, specifically within the {domain} domain."

async def generate_entity_types(
        client: AsyncOpenAI,
        model_name: str,
        domain: str,
        persona: str,
        docs: List[str], # 输入现在是文档块列表
        task: str = DEFAULT_TASK,
        json_mode: bool = False,
) -> List[str]:
    """使用 LLM API 从给定的文档块中生成实体类型"""
    if not docs:
        print("没有提供任何文档块，跳过实体类型生成。")
        return []
    # (此函数内部逻辑与之前代码完全相同，此处省略)
    if not BAILIAN_API_KEY:
        print("错误: 环境变量 DASHSCOPE_API_KEY 未设置。")
        return []
    formatted_task = task.format(domain=domain)
    docs_str = "\n---\n".join(docs) # 使用分隔符连接文档块
    prompt = (
        ENTITY_TYPE_GENERATION_JSON_PROMPT if json_mode else ENTITY_TYPE_GENERATION_PROMPT
    ).format(task=formatted_task, input_text=docs_str)
    messages = [{"role": "system", "content": persona}, {"role": "user", "content": prompt}]
    try:
        if json_mode:
            response = await client.chat.completions.create(
                model=model_name, messages=messages, response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            try:
                parsed_model = EntityTypesResponse.model_validate_json(content)
                return parsed_model.entity_types
            except (ValidationError, json.JSONDecodeError) as e:
                print(f"错误: JSON 解析或验证失败。错误: {e}\n原始输出: {content}")
                return []
        else:
            response = await client.chat.completions.create(model=model_name, messages=messages)
            content = response.choices[0].message.content
            if content:
                return [item.strip() for item in content.split(',')]
            return []
    except APIError as e:
        print(f"错误: 调用 API 失败。状态码: {e.status_code}, 响应: {e.response}")
        return []
    except Exception as e:
        print(f"发生未知错误: {e}")
        return []


# --- 5. 整合后的主工作流 ---
def readfile(file_path: str) -> List[str]:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file if line.strip()]
            return lines
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到。")
        return []
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return []

async def main_workflow():
    """完整的工作流: 文档选择 -> 实体生成"""

    # 原始文档，可以更长更复杂
    raw_documents = readfile(r"dataset/processed_text_070801.txt")

    print("="*50)
    print("启动文档分析工作流")
    print("="*50)

    # --- 步骤 1: 使用 Ollama 选择代表性文档块 ---
    print("\n--- 步骤 1: 文档预处理与选择 ---")
    embedding_model = OllamaEmbeddingModel(model_name=OLLAMA_EMBEDDING_MODEL)
    if not embedding_model.client:
        return # 如果无法连接Ollama，则终止

    selected_docs = await load_docs_with_auto_selection(
        documents=raw_documents,
        embedding_model=embedding_model,
        k=40,  # 选择8个最具代表性的块
        chunk_size=800,
        overlap=100
    )

    if not selected_docs:
        print("未能选择任何文档块，工作流终止。")
        return

    print("\n--- 已选择的核心文档块 ---")
    for i, doc in enumerate(selected_docs, 1):
        print(f"[{i}] {doc}")

    # --- 步骤 2: 使用百炼 LLM 生成实体类型 ---
    print("\n--- 步骤 2: 生成实体类型 ---")
    llm_client = AsyncOpenAI(api_key=BAILIAN_API_KEY, base_url=BAILIAN_API_BASE_URL)
    domain = "Structural Analysis & Structural Mechanism"
    persona = "You are an expert analyst skilled in identifying and categorizing entities from text documents."

    # 同时运行文本和 JSON 模式
    text_task = generate_entity_types(
        client=llm_client, model_name=BAILIAN_MODEL_NAME, domain=domain, persona=persona, docs=selected_docs, json_mode=False
    )
    json_task = generate_entity_types(
        client=llm_client, model_name=BAILIAN_MODEL_NAME, domain=domain, persona=persona, docs=selected_docs, json_mode=True
    )

    text_results, json_results = await asyncio.gather(text_task, json_task)

    print("\n--- 分析结果 ---")
    print(f"文本模式输出: {text_results}")
    print(f"JSON 模式输出: {json_results}")
    print("="*50)


if __name__ == "__main__":
    # 确保在Windows上能正常运行asyncio
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main_workflow())