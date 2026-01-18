import os
import re
import asyncio
import json
from pydantic import BaseModel
from aiofiles import open as aio_open

# 导入 LLM 模型
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate

# 工具函数：tiktoken 编码与解码
from doke_rag.core.utils import (
    compute_mdhash_id,
    logger,
    encode_string_by_tiktoken,
    decode_tokens_by_tiktoken
)

from datetime import datetime
from typing import Any, List
from pathlib import Path
from doke_rag.core import LightRAG
from doke_rag.core.base import DocStatus
from doke_rag.config.paths import RAW_DATA_DIR

# 定义 Pydantic 数据模型（结构化输出）
class Sentences(BaseModel):
    sentences: List[str]

# 定义 prompt 模板
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Please process the input technical text according to the instructions below to generate a list of structured and self-contained knowledge statements:

1. **Sentence Simplification**: Split long or complex sentences into shorter, clear ones. Each sentence must have an explicit subject and preserve the original terminology as much as possible.
2. **Technical Term Substitution**: Replace vague pronouns (e.g., "it", "this", "that") with the full technical terms they refer to, so each sentence is context-independent and self-contained.
3. **Equation Handling**:
   - Retain all mathematical expressions and equation numbers exactly as they appear.
   - If a sentence explains a formula (e.g., Eq. 11-8), combine the formula reference and explanation into one clear statement.
   - **Never fabricate** equation numbers or contents.
4. **Figure Handling**:
   - Preserve any Markdown-style figure links as they appear in the input, along with their associated captions or references like "Fig. 11-8 a and b".
   - Do not describe or generate figure content. Do not modify or invent figure references.
5. **Logical Structuring**: Group related sentences into logical knowledge points; re-organize only when it improves clarity without changing the meaning.
6. **Output Format**: The final result must be a valid JSON list, where each element is a self-contained knowledge sentence as a string.
7. **Do Not Include**: Any examples, illustrations, explanations meant for teaching or demonstration purposes unless they define core concepts.

### Example (Do NOT use this as input)
#### Input:
In structural analysis, the force method is used to solve statically indeterminate structures. The idea is to transform an indeterminate structure into a determinate one. For example, as shown in the figure...

#### Output:
[
  "The force method is used to solve statically indeterminate structures by transforming them into determinate forms.",
  "A basic system is created by removing redundant supports and introducing unknown forces, allowing the use of the force method."
]

## Reminder: Do not include the word 'Example' or any figures or equations unless they are explicitly referenced by number or Markdown syntax in the input.
        """,
    ),
    ("human", "Please process the following input text accordingly:\n{input}")
])

def chunking_by_token_size(
    content: str,
    overlap_token_size: int = 800,  # 保留 200 tokens，避免语义丢失
    max_token_size: int = 5000,       # 限制 chunk 最大 1000 tokens
    tiktoken_model: str = "gpt-4o",
) -> list[dict[str, Any]]:
    """
    基于 token 长度采用滑动窗口进行分块，确保不会在句子中间截断。
    先按标点符号切分，再以句子为单位累计 token 数，超过 max_token_size 时保存当前 chunk，
    并使用 overlap_token_size 作为滑动窗口保留一部分上下文。
    """
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results: list[dict[str, Any]] = []
    sentences = re.split(r'(?<=[.!?])\s+', content)
    new_chunks = []
    current_chunk = []
    current_token_count = 0

    for sentence in sentences:
        sentence_tokens = encode_string_by_tiktoken(sentence, model_name=tiktoken_model)
        sentence_length = len(sentence_tokens)

        if current_token_count + sentence_length > max_token_size:
            chunk_text = " ".join(current_chunk).strip()
            chunk_tokens = encode_string_by_tiktoken(chunk_text, model_name=tiktoken_model)
            new_chunks.append((len(chunk_tokens), chunk_text))
            # 保留部分 tokens 作为下一段的上下文
            overlap_text = decode_tokens_by_tiktoken(chunk_tokens[-overlap_token_size:], model_name=tiktoken_model)
            current_chunk = [overlap_text]
            current_token_count = len(encode_string_by_tiktoken(overlap_text, model_name=tiktoken_model))
        current_chunk.append(sentence)
        current_token_count += sentence_length

    if current_chunk:
        chunk_text = " ".join(current_chunk).strip()
        chunk_tokens = encode_string_by_tiktoken(chunk_text, model_name=tiktoken_model)
        new_chunks.append((len(chunk_tokens), chunk_text))

    for index, (_len, chunk) in enumerate(new_chunks):
        results.append({
            "tokens": _len,
            "content": chunk.strip(),
            "chunk_order_index": index,
        })
    return results

def process_with_prompt(llm, prompt, text):
    """
    使用结构化输出模型 + prompt 同步调用 LLM 并获取 JSON 格式结构化输出。
    """
    try:
        structured_llm = llm.with_structured_output(schema=Sentences)
        runnable = prompt | structured_llm
        result = runnable.invoke({"input": text})
        return result.sentences if result else []
    except Exception as e:
        print(f"[❌ Error] Failed to process chunk: {e}")
        return []

def get_llm(llm_source: str = "tongyi"):
    """
    根据 llm_source 参数返回相应的 LLM 实例。
    可选值：
      - "tongyi"：使用 ChatTongyi 模型；
      - "openai"：使用 ChatOpenAI 模型。
    """
    if llm_source == "tongyi":
        return ChatTongyi(
            model="qwen-plus",#qwq-plus  /  qwen-max-latest
            api_key=os.getenv("ALIYUN_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0,
            streaming=True,
        )
    elif llm_source == "openai":
        return ChatOpenAI(
            model="qwen2.5:7b",
            openai_api_key="ollama",
            base_url="http://localhost:11434/v1",
            temperature=0
        )
    else:
        raise ValueError(f"Unsupported LLM source: {llm_source}")

def split_into_structured_sentences(text: str, llm_source: str = "tongyi"):
    llm = get_llm(llm_source)
    if text:
        try:
            print("Processing chunk...")
            processed_text = process_with_prompt(llm, prompt, text)
            print("Processed Text:", processed_text)
            return processed_text
        except Exception as e:
            print(f"[❌ Error] Failed to process chunk: {e}")
            return []  # 或者 return None

async def split_into_structured_sentences_async(text: str, llm_source: str = "tongyi"):
    """
    异步包装 run_chunk，将同步的 LLM 调用放入线程池中执行。
    """
    return await asyncio.to_thread(split_into_structured_sentences, text, llm_source)

async def store_processed_text(results: list, output_file: str):
    """
    将所有 processed_text（处理结果）存储到指定文件中，文件格式为 JSON。
    """
    async with aiofiles.open(output_file, "w", encoding="utf-8") as f:
        await f.write(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Processed texts stored to {output_file}")

async def main():
    """
    主函数：
      - 异步读取数据文件（使用 aiofiles）；
      - 对每个文件进行 token 分块；
      - 对每个块并发调用 LLM 处理；
      - 最后将所有的 processed_text 存储到文件中。
    """

    # 使用相对路径，数据目录相对于项目根目录
    datapath = RAW_DATA_DIR / "texts"
    input_files = [file for file in os.listdir(datapath) if file.endswith(('.txt', '.md'))]
    tasks = []
    for input_file in input_files:
        file_path = os.path.join(datapath, input_file)
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            input_text = await f.read()
        chunks = chunking_by_token_size(input_text)
        for chunk in chunks:
            # 可通过修改 llm_source 参数来切换不同的 LLM（例如 "tongyi" 或 "openai"）
            tasks.append(split_into_structured_sentences_async(chunk["content"], llm_source="tongyi"))
    # 并发执行所有任务
    results = await asyncio.gather(*tasks)
    print("All tasks completed.")
    # 将所有 processed_text 存储到文件中
    await store_processed_text(results, "processed_texts_0708.json")


# 继承 LightRAG 并重写预处理方法，增加分块和并行调用 run_chunk_async
class LightRAGWithChunking(LightRAG):
    async def apipeline_enqueue_documents_with_chunking(
        self, input: str | list[str], ids: list[str] | None = None
    ) -> None:
        """
        处理文档并加入队列（包含分块 & run_chunk 预处理）
        """
        # 1️⃣ 清理 & 去重
        if isinstance(input, str):
            input = [input]
        if isinstance(ids, str):
            ids = [ids]

        if ids is not None:
            if len(ids) != len(input):
                raise ValueError("Number of IDs must match the number of documents")
            if len(ids) != len(set(ids)):
                raise ValueError("IDs must be unique")
            contents = {id_: doc for id_, doc in zip(ids, input)}
        else:
            input = list(set(self.clean_text(doc) for doc in input))  # 清理 & 去重
            contents = {compute_mdhash_id(doc, prefix="temp-"): doc for doc in input}

        # 2️⃣ 对文本进行分块
        chunked_contents = {}
        for temp_doc_id, content in contents.items():
            chunks = chunking_by_token_size(content)  # 分块
            chunked_contents[temp_doc_id] = [chunk["content"] for chunk in chunks]

        # 3️⃣ 并行调用 run_chunk_async 进行预处理
        processed_contents = {}
        for temp_doc_id, chunks in chunked_contents.items():
            processed_chunks = await asyncio.gather(*[split_into_structured_sentences_async(chunk, llm_source="tongyi") for chunk in chunks])
            # 这里假设 run_chunk_async 返回的是一个列表（例如句子列表）
            processed_text = "\n".join(["\n".join(chunk) for chunk in processed_chunks if isinstance(chunk, list)])
            final_doc_id = compute_mdhash_id(processed_text, prefix="doc-")
            processed_contents[final_doc_id] = processed_text

        # 4️⃣ 构建新文档结构
        new_docs: dict[str, Any] = {
            doc_id: {
                "content": processed_contents[doc_id],
                "content_summary": self._get_content_summary(processed_contents[doc_id]),
                "content_length": len(processed_contents[doc_id]),
                "status": DocStatus.PENDING,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
            for doc_id in processed_contents
        }

        # 5️⃣ 过滤重复文档
        all_new_doc_ids = set(new_docs.keys())
        unique_new_doc_ids = await self.doc_status.filter_keys(all_new_doc_ids)

        ignored_ids = [doc_id for doc_id in all_new_doc_ids if doc_id not in unique_new_doc_ids]
        if ignored_ids:
            logger.warning(f"Ignoring {len(ignored_ids)} document IDs not found in new_docs")
            for doc_id in ignored_ids:
                logger.warning(f"Ignored document ID: {doc_id}")

        # 6️⃣ 只存储唯一的新文档
        new_docs = {doc_id: new_docs[doc_id] for doc_id in unique_new_doc_ids if doc_id in new_docs}

        if not new_docs:
            logger.info("No new unique documents were found.")
            return

        await self.doc_status.upsert(new_docs)  # 存储数据
        logger.info(f"Stored {len(new_docs)} new unique documents after chunking")
if __name__ == "__main__":
    asyncio.run(main())