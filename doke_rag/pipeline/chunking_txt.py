import os
import re
import asyncio
import aiofiles
import time
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate

from doke_rag.core.utils import (
    encode_string_by_tiktoken,
    decode_tokens_by_tiktoken
)
from doke_rag.config.paths import RAW_DATA_DIR

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
你是一名技术助理，负责处理土木工程与结构力学文本。

你的任务是从输入的学术内容中提取出干净、逻辑清晰的知识点。请遵循以下规则：

---

### ✳️ 处理规则：

1. **句子简化**：

   * 将冗长或复杂的句子拆分为更短、更清晰的句子。
   * 每个句子必须包含明确的主语，并保留技术术语。

2. **代词替换**：

   * 将含糊的代词（如“it”“this”“that”）替换为其具体所指，以确保句子独立且清晰。

3. **公式处理**：

   * 保留所有数学表达式和公式编号 **与原文完全一致**（例如 `Eq. 11-8`）。
   * 将公式的解释与其引用结合在一个句子中。
   * ❌ 不得编造或修改任何公式或公式编号。

4. **图示处理**：

   * 保留所有图示引用（例如 “Fig. 11-8 a and b”）以及任何 Markdown 格式的图片链接（如 `![Fig. 11-8 a and b](...)`）**与原文完全一致**。
   * 不要描述或解释图示内容。
   * ❌ 不得删除或虚构图号或图片链接。

5. **逻辑组织**：

   * 将相关句子归纳为清晰的知识块。
   * 仅在不改变技术含义的前提下，对内容进行重组以提升清晰度。

6. **输出格式**：

   * 输出必须为干净、可读的英文段落式文本。

---

### 示例（⚠️ 不要将此作为输入）

#### 输入：

Eq. 5-4, \$M = EI \frac{{d^2y}}{{dx^2}}\$, describes the relation between bending moment and curvature. It is used at both ends of the beam shown in Fig. 5-4a and b.
![Fig. 5-4a and b](https://example.com/fig5-4.png)

#### 输出：

Eq. 5-4, \$M = EI \frac{{d^2y}}{{dx^2}}\$, expresses the relationship between bending moment and curvature.
Eq. 5-4 is applied at both ends of the beam shown in Fig. 5-4a and b.
![Fig. 5-4a and b](https://example.com/fig5-4.png)

---

现在请对下面的输入做同样的处理：

        """
#         """
# You are a technical assistant processing civil engineering and structural mechanics texts.
#
# Your task is to extract clean, logically structured knowledge points from the input academic content. Follow the rules below:
#
# ---
#
# ### ✳️ Processing Rules:
#
# 1. **Sentence Simplification**:
#    - Break long or complex sentences into shorter, clearer ones.
#    - Each sentence must include an explicit subject and retain technical terms.
#
# 2. **Pronoun Replacement**:
#    - Replace vague pronouns (like "it", "this", "that") with their specific referents to ensure standalone clarity.
#
# 3. **Equation Handling**:
#    - Preserve all mathematical expressions and equation numbers **exactly as they appear** (e.g., `Eq. 11-8`).
#    - Combine any explanation of an equation with its reference in a single sentence.
#    - ❌ Never fabricate or modify any equations or equation numbers.
#
# 4. **Figure Handling**:
#    - Keep all references to figures (e.g., "Fig. 11-8 a and b") and any Markdown-style image links like `![Fig. 11-8 a and b](...)` **exactly as in the input**.
#    - Do **not describe** or interpret figure contents.
#    - ❌ Never delete or invent figure numbers or image URLs.
#
# 5. **Logical Organization**:
#    - Group related sentences into clear knowledge blocks.
#    - Only reorganize content if it improves clarity without changing the technical meaning.
#
# 6. **Output Format**:
#    - Output must be a clean, readable English paragraph-style text.
#
# ---
#
# ### Example (Do NOT use this as input)
# #### Input:
# Eq. 5-4, $M = EI \\frac{{d^2y}}{{dx^2}}$, describes the relation between bending moment and curvature. It is used at both ends of the beam shown in Fig. 5-4a and b.
# ![Fig. 5-4a and b](https://example.com/fig5-4.png)
#
# #### Output:
# Eq. 5-4, $M = EI \\frac{{d^2y}}{{dx^2}}$, expresses the relationship between bending moment and curvature.
# Eq. 5-4 is applied at both ends of the beam shown in Fig. 5-4a and b.
# ![Fig. 5-4a and b](https://example.com/fig5-4.png)
#
# ---
#
# Now please apply the same rules to the following input:
#         """
    ),
    ("human", "Text:\n{input}")
])


def chunking_by_token_size(
    content: str,
    overlap_token_size: int = 600,
    max_token_size: int = 3000,
    tiktoken_model: str = "gpt-4o"
) -> list[dict]:
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
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

def get_llm(llm_source: str = "tongyi"):
    if llm_source == "tongyi":
        return ChatTongyi(
            model="deepseek-r1",
            api_key=os.getenv("ALIYUN_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0,
            streaming=False,
        )
    elif llm_source == "openai":
        return ChatOpenAI(
            model="gpt-4o",
            openai_api_key="sk-xxx",
            base_url="https://api.openai.com/v1",
            temperature=0
        )
    else:
        raise ValueError(f"Unsupported LLM source: {llm_source}")

def process_text_chunk(text: str, llm_source: str = "tongyi") -> str:
    for attempt in range(3):
        try:
            llm = get_llm(llm_source)
            runnable = prompt | llm
            result = runnable.invoke({"input": text})
            return result.content.strip()
        except Exception as e:
            print(f"[❌ Error] Attempt {attempt+1} failed: {e}")
            time.sleep(15)
    return ""


# === 异步封装 ===
async def process_text_chunk_async(text: str, llm_source: str = "tongyi") -> str:
    return await asyncio.to_thread(process_text_chunk, text, llm_source)

# === 保存为 txt ===
async def save_to_txt(results: list[str], output_path: str):
    content = "\n\n---\n\n".join(results)
    async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
        await f.write(content)
    print(f"[✅] Output saved to {output_path}")

# === 主流程 ===
async def main():
    # 使用相对路径，数据目录相对于项目根目录
    datapath = RAW_DATA_DIR / "texts"
    output_path = RAW_DATA_DIR / "split_en2zh.txt"
    input_files = [f for f in os.listdir(datapath) if f.endswith((".txt", ".md"))]

    all_tasks = []

    for file in input_files:
        file_path = os.path.join(datapath, file)
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            raw_text = await f.read()
        chunks = chunking_by_token_size(raw_text)
        for chunk in chunks:
            all_tasks.append(process_text_chunk_async(chunk["content"], llm_source="tongyi"))

    print(f"[⚙️] Processing {len(all_tasks)} chunks...")
    results = await asyncio.gather(*all_tasks)
    await save_to_txt(results, output_path)

if __name__ == "__main__":
    asyncio.run(main())
