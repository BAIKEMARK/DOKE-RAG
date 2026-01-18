import os
import json
import requests
import numpy as np
import csv
import re
import time
import Levenshtein
from openai import OpenAI, AzureOpenAI
from dotenv import load_dotenv
from sklearn.cluster import DBSCAN
from typing import List, Dict, Any, Tuple
from pathlib import Path

from doke_rag.config.paths import WORKING_DIR

load_dotenv()
# ==============================================================================
# 基础组件
# ==============================================================================

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """计算两个Numpy向量的余弦相似度"""
    if not isinstance(vec1, np.ndarray) or not isinstance(vec2, np.ndarray) or vec1.size == 0 or vec2.size == 0:
        return 0.0
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return np.dot(vec1, vec2) / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0

class SlidingWindowSimilarity:
    """滑动窗口相似度计算类，用于解决长短文本相似度不平衡问题"""
    def __init__(self, length_ratio_threshold: float = 2.0, overlap_ratio: float = 0.3):
        self.length_ratio_threshold = length_ratio_threshold
        self.overlap_ratio = overlap_ratio

    def _create_sliding_windows(self, text: str, window_size: int) -> List[str]:
        if window_size <= 0 or window_size >= len(text):
            return [text]
        stride = int(window_size * (1 - self.overlap_ratio)) or 1
        windows = [text[i:i + window_size] for i in range(0, len(text) - window_size + 1, stride)]
        if windows and text and windows[-1][-1] != text[-1]:
            windows.append(text[-window_size:])
        return windows

    def calculate_similarity_with_cache(self, text1: str, text2: str, embedding_func: callable, embedding_cache: dict) -> Tuple[float, dict]:
        """使用缓存计算两个文本的相似度，避免重复计算嵌入向量"""
        if not text1 or not text2:
            return (1.0 if not text1 and not text2 else 0.0), embedding_cache

        def get_embed_from_cache(txt):
            if txt not in embedding_cache:
                embedding_cache[txt] = embedding_func(txt)
            return embedding_cache[txt]

        len1, len2 = len(text1), len(text2)
        if min(len1, len2) == 0 or max(len1, len2) / min(len1, len2) < self.length_ratio_threshold:
            sim = cosine_similarity(get_embed_from_cache(text1), get_embed_from_cache(text2))
            return sim, embedding_cache

        long_text, short_text = (text1, text2) if len1 > len2 else (text2, text1)
        short_vec = get_embed_from_cache(short_text)
        windows = self._create_sliding_windows(long_text, len(short_text))

        max_sim = 0.0
        for window in windows:
            window_vec = get_embed_from_cache(window)
            max_sim = max(max_sim, cosine_similarity(short_vec, window_vec))
        return max_sim, embedding_cache


# ==============================================================================
# 优化后的实体去重主类
# ==============================================================================

class OptimizedEntityDeduplicator:
    """
    经优化的实体去重解决方案
    工作流程: DBSCAN初筛 -> 簇内精细比较 -> 连通分量预分组 -> LLM最终验证
    """

    def __init__(self,
                 similarity_threshold: float = 0.75,
                 name_weight: float = 0.2,
                 content_weight: float = 0.8,
                 model_name: str = "quentinz/bge-large-zh-v1.5",
                 ollama_base_url: str = "http://localhost:11434",
                 llm_provider: str = 'ollama',  # 新增: LLM服务提供商 ('ollama' or 'aliyun')
                 llm_model: str = "deepseek-r1-0528",
                 llm_batch_size: int = 5,
                 aliyun_api_key: str = None,  # 新增: 阿里云百炼API Key
                 aliyun_api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1", # 新增: 阿里云百炼API Endpoint
                 dbscan_eps: float = 0.2,
                 dbscan_min_samples: int = 2,
                 length_ratio_threshold: float = 1.5,
                 overlap_ratio: float = 0.3):

        self.similarity_threshold = similarity_threshold
        self.name_weight = name_weight
        self.content_weight = content_weight
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url

        # --- 新增和修改的LLM配置 ---
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.aliyun_api_key = aliyun_api_key
        self.aliyun_api_base = aliyun_api_base

        self.llm_batch_size = llm_batch_size
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.sliding_window = SlidingWindowSimilarity(length_ratio_threshold, overlap_ratio)
        self.embedding_cache = {}

        # --- 配置验证 ---
        if self.llm_provider == 'aliyun' and not self.aliyun_api_key:
            raise ValueError("当 'llm_provider' 设置为 'aliyun' 时, 'aliyun_api_key' 不能为空。")

    def get_embedding(self, text: str) -> np.ndarray:
        """获取文本的嵌入向量"""
        if not text: return np.array([])
        try:
            # 检查缓存，避免重复API调用
            if text in self.embedding_cache:
                return self.embedding_cache[text]
            response = requests.post(f"{self.ollama_base_url}/api/embeddings", json={"model": self.model_name, "prompt": text})
            response.raise_for_status()
            embedding = np.array(response.json().get("embedding", []))
            self.embedding_cache[text] = embedding # 存入缓存
            return embedding
        except Exception as e:
            print(f"获取嵌入向量时出错: {e}")
            return np.array([])

    def _get_prop(self, entity: Dict, key: str, default: Any = "") -> Any:
        return entity.get(key, entity.get(f"__{key}__", entity.get(f"entity_{key}", default)))

    def load_entities(self, file_path: str) -> List[Dict]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get("data", []) if isinstance(data, dict) else data
        except Exception as e:
            print(f"加载实体数据时出错: {e}"); return []

    def _verify_groups_with_llm(self, candidate_groups: List[List[Dict]]) -> Dict[Tuple[str, str], Tuple[bool, str]]:
        """对预分组进行LLM验证，支持OpenAI、Azure OpenAI、阿里云百炼（统一SDK风格）"""
        pairs_to_verify = []
        for group in candidate_groups:
            representative = max(group, key=lambda e: len(self._get_prop(e, 'content')))
            for member in group:
                if self._get_prop(member, 'id') != self._get_prop(representative, 'id'):
                    pairs_to_verify.append({"entity1": representative, "entity2": member})

        print(f"[LOG] 共生成 {len(pairs_to_verify)} 个实体对交由LLM ({self.llm_provider}) 进行最终判断...")
        if not pairs_to_verify:
            return {}

        verified_results = []

        for i in range(0, len(pairs_to_verify), self.llm_batch_size):
            batch = pairs_to_verify[i:i + self.llm_batch_size]

            system_prompt = """You are an authoritative expert in the field of structural mechanics, proficient in mechanical principles, concept analysis, and terminology standardization. Please carefully determine if each pair of entities below refers to the same mechanical concept in essence, strictly adhering to the following standards and instructions:

[Basic Judgment Principles]
1. The core judgment must focus on **mechanical principles, fundamental definitions, and essential mechanisms of action**, rather than merely on the similarity of names, abbreviations, or syntactic expressions.
2. For **synonyms, linguistic variants, or near-synonymous expressions**, they should be judged as "same" only if they are completely identical in physical meaning, calculation basis, and application context.
3. If two entities use different expressions but reveal the exact same fundamental principle or definition (i.e., different explanatory paths describing the same core mechanism), they are considered "same".
4. If one entity is a **sub-concept, special case, specific application scenario, or derived result** of another (such as local variables, auxiliary parameters, etc.), it must be strictly judged as "not same".
5. When two entities have essential differences in **physical dimensions, definition methods, calculation basis, or the objects they act upon**, they must be judged as "not same", even if their names or parts of their descriptions are similar.
6. For mathematical symbols (e.g., δ11, Δ1P), they are considered "same" only when they explicitly refer to the identical physical meaning.

[Additional Strict Requirements]
- You must be extremely cautious when making a "same" (true) judgment. Only output "true" after a detailed comparison confirms that the two entities are completely consistent in their core concepts, calculation basis, and application context.
- In cases of doubt or borderline examples, always make a conservative judgment of "not same" (false).

[Output Requirements]
1. Analyze and judge each pair of entities independently.
2. Strictly output a JSON object in the following format. **Each pair must be a separate item**.
3. The JSON keys must be "pair_1", "pair_2", and so on.
4. Each item must include:
   - `is_same`: A boolean value, true or false.
   - `reason`: A concise and clear justification for the judgment (under 20 words).
```json
{
  "pair_1": {
    "is_same": true/false,
    "reason": "A brief analysis of the reason (under 20 words)."
  },
  "pair_2": {
    "is_same": true/false,
    "reason": "A brief analysis of the reason (under 20 words)."
  }
}```
[Other Important Notes]

The output must consist solely of the JSON object described above, without any additional text or merged items.

The judgment for each pair must be based on rigorous and detailed analysis and reasoning, ensuring sufficient evidence. Use "true" only when the core concepts described are confirmed to be identical.

If there is any ambiguity or uncertainty during the judgment process, conservatively output "false" to avoid misclassification.

All reasons must be concise, directly address the key differences, and be kept under 20 words.

Before outputting, check if your output conforms to the [Output Requirements]. If not, regenerate the output according to the requirements.

Here are the entity pairs to be analyzed:
\n\n
"""

            user_content = "以下为待分析的实体对：\n\n"
            for j, pair in enumerate(batch):
                user_content += f"## 实体对 {j + 1}\n实体1 - {self._get_prop(pair['entity1'], 'name')}:\n{self._get_prop(pair['entity1'], 'content')}\n\n实体2 - {self._get_prop(pair['entity2'], 'name')}:\n{self._get_prop(pair['entity2'], 'content')}\n\n"

            try:
                # -------- 初始化客户端 --------
                if self.llm_provider == "openai":
                    client = OpenAI(
                        api_key=self.openai_api_key,
                        base_url=self.openai_api_base
                    )
                elif self.llm_provider == "azure":
                    client = AzureOpenAI(
                        api_key=self.azure_api_key,
                        api_version="2024-02-01",
                        azure_endpoint=self.azure_api_base
                    )
                elif self.llm_provider == "aliyun":
                    client = OpenAI(
                        api_key=self.aliyun_api_key,
                        base_url=self.aliyun_api_base
                    )
                else:
                    raise ValueError(f"不支持的 LLM provider: {self.llm_provider}")

                # -------- 调用 Chat Completions API --------
                completion = client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0
                )

                llm_response_text = completion.choices[0].message.content

                # -------- 解析 LLM 响应 --------
                try:
                    response_data = json.loads(llm_response_text)
                except json.JSONDecodeError:
                    json_match = re.search(r"```json\s*([\s\S]+?)\s*```", llm_response_text)
                    if not json_match:
                        raise ValueError("LLM响应中未找到有效的JSON代码块。")
                    response_data = json.loads(json_match.group(1))

                for j, pair in enumerate(batch):
                    pair_key = f"pair_{j+1}"
                    ids_tuple = tuple(sorted((self._get_prop(pair['entity1'], 'id'),
                                              self._get_prop(pair['entity2'], 'id'))))
                    if pair_key in response_data:
                        is_same = response_data[pair_key].get("is_same", False)
                        reason = response_data[pair_key].get("reason", "无理由提供")
                        verified_results.append((ids_tuple, (is_same, reason)))

            except Exception as e:
                print(f"LLM批处理验证出错: {e}")
                for pair in batch:
                    ids_tuple = tuple(sorted((self._get_prop(pair['entity1'], 'id'),
                                              self._get_prop(pair['entity2'], 'id'))))
                    verified_results.append((ids_tuple, (False, f"LLM请求错误: {e}")))

            if i + self.llm_batch_size < len(pairs_to_verify):
                time.sleep(1)

        # 转为 dict 返回
        return dict(verified_results)

    def run_deduplication(self, input_file: str, output_prefix: str, output_dir: str = "dedup_outputs"):
        """执行完整的、经过优化的实体去重流程"""
        # === 步骤 0: 加载数据 ===
        print(f"[LOG] 步骤 0: 从 {input_file} 加载实体...")
        entities = self.load_entities(input_file)
        if not entities or len(entities) < 2:
            print("[LOG] 实体数量不足，无需去重。"); return
        all_entities_map = {self._get_prop(e, 'id'): e for e in entities}
        print(f"[LOG] 加载了 {len(entities)} 个实体。")

        # === 步骤 1: DBSCAN 聚类 ===
        print("\n[LOG] 步骤 1: 预计算实体内容向量并执行DBSCAN聚类...")
        embeddings = np.array([self.get_embedding(self._get_prop(e, "content")) for e in entities])

        # 过滤掉空的或无效的嵌入向量
        valid_indices = [i for i, emb in enumerate(embeddings) if emb.any()]
        if len(valid_indices) < 2:
            print("[LOG] 有效实体数量不足，无法进行聚类。"); return

        valid_embeddings = embeddings[valid_indices]
        db = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples, metric='cosine').fit(valid_embeddings)

        clusters = {}
        for i, label in enumerate(db.labels_):
            if label != -1: # -1是噪声点，不处理
                original_index = valid_indices[i]
                clusters.setdefault(label, []).append(original_index)
        print(f"[LOG] DBSCAN完成，发现 {len(clusters)} 个候选簇。")

        # === 步骤 2: 簇内精细相似度计算 ===
        print("\n[LOG] 步骤 2: 在簇内部计算实体对的综合相似度...")
        similar_pairs_above_threshold = []
        for cluster_indices in clusters.values():
            if len(cluster_indices) < 2: continue
            for i in range(len(cluster_indices)):
                for j in range(i + 1, len(cluster_indices)):
                    e1 = entities[cluster_indices[i]]
                    e2 = entities[cluster_indices[j]]

                    name1, name2 = self._get_prop(e1, "name"), self._get_prop(e2, "name")
                    content1, content2 = self._get_prop(e1, "content"), self._get_prop(e2, "content")

                    name_sim = 1 - Levenshtein.distance(name1, name2) / max(len(name1), len(name2)) if max(len(name1), len(name2)) > 0 else 1.0
                    content_sim, self.embedding_cache = self.sliding_window.calculate_similarity_with_cache(content1, content2, self.get_embedding, self.embedding_cache)
                    combined_sim = self.name_weight * name_sim + self.content_weight * content_sim

                    if combined_sim >= self.similarity_threshold:
                        similar_pairs_above_threshold.append({"entity1": e1, "entity2": e2})
        print(f"[LOG] 找到 {len(similar_pairs_above_threshold)} 对相似度高于阈值的实体对。")

        # === 步骤 3: 预分组 (计算连通分量) ===
        if not similar_pairs_above_threshold:
            print("[LOG] 未发现需要合并的实体组，流程结束。"); return

        print("\n[LOG] 步骤 3: 基于相似度图计算连通分量，形成候选合并组...")
        graph = {}
        for pair in similar_pairs_above_threshold:
            id1, id2 = self._get_prop(pair["entity1"], "id"), self._get_prop(pair["entity2"], "id")
            graph.setdefault(id1, set()).add(id2)
            graph.setdefault(id2, set()).add(id1)

        visited, candidate_groups = set(), []
        for node in graph:
            if node not in visited:
                component, queue = [], [node]
                visited.add(node)
                head = 0
                while head < len(queue):
                    current_id = queue[head]; head += 1
                    if current_id in all_entities_map:
                        component.append(all_entities_map[current_id])
                    for neighbor_id in graph.get(current_id, set()):
                        if neighbor_id not in visited:
                            visited.add(neighbor_id); queue.append(neighbor_id)
                if len(component) > 1: candidate_groups.append(component)
        print(f"[LOG] 形成 {len(candidate_groups)} 个候选合并组。")

        # === 步骤 4: LLM 对预分组进行验证 ===
        print("\n[LOG] 步骤 4: 对每个候选组进行LLM最终验证...")
        llm_results_map = self._verify_groups_with_llm(candidate_groups)

        # === 步骤 5: 生成最终结果并保存 ===
        print("\n[LOG] 步骤 5: 整理最终结果并保存文件...")
        final_merged_groups = []
        detailed_log_data = []

        for group in candidate_groups:
            representative = max(group, key=lambda e: len(self._get_prop(e, 'content')))
            rep_id = self._get_prop(representative, 'id')
            rep_name = self._get_prop(representative, 'name')

            final_to_merge_names = []
            for member in group:
                mem_id = self._get_prop(member, 'id')
                mem_name = self._get_prop(member, 'name')
                if mem_id == rep_id: continue

                ids_tuple = tuple(sorted((rep_id, mem_id)))
                is_same, reason = llm_results_map.get(ids_tuple, (False, "未进行LLM验证"))

                detailed_log_data.append({
                    "representative_name": rep_name,
                    "member_name": mem_name,
                    "llm_verified_same": is_same,
                    "llm_reason": reason
                })
                if is_same:
                    final_to_merge_names.append(mem_name)

            if final_to_merge_names:
                final_merged_groups.append({"representative": rep_name, "to_merge": final_to_merge_names})

        # 保存详细日志CSV
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"{output_prefix}_details.csv")
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=["representative_name", "member_name", "llm_verified_same", "llm_reason"])
                writer.writeheader()
                writer.writerows(detailed_log_data)
            print(f"[SUCCESS] 详细验证日志已保存到: {csv_path}")
        except Exception as e:
            print(f"[ERROR] 保存CSV文件时出错: {e}")

        # 保存最终合并JSON
        json_path = os.path.join(output_dir, f"{output_prefix}_merged.json")
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(final_merged_groups, f, ensure_ascii=False, indent=2)
            print(f"[SUCCESS] 最终合并建议已保存到: {json_path}")
        except Exception as e:
            print(f"[ERROR] 保存JSON文件时出错: {e}")

        print(f"\n[SUCCESS] 去重流程完成，发现 {len(final_merged_groups)} 组可合并实体。")

# ==============================================================================
# 使用示例
# ==============================================================================
if __name__ == "__main__":

    # --- 选择LLM服务商: 'ollama' 或 'aliyun' ---
    LLM_PROVIDER_CHOICE = 'aliyun'  # <-- 在这里切换

    # --- 统一配置 ---
    config = {
        # 基础参数
        "similarity_threshold": 0.75,
        "name_weight": 0.2,
        "content_weight": 0.8,
        "length_ratio_threshold": 1.3,
        "overlap_ratio": 0.3,
        # Embedding模型参数
        "model_name": "nomic-embed-text",
        "ollama_base_url": "http://localhost:11434",
        # DBSCAN 参数
        "dbscan_eps": 0.15,
        "dbscan_min_samples": 4,
        # LLM 参数
        "llm_provider": LLM_PROVIDER_CHOICE,
        "llm_model": "deepseek-r1-0528", # 模型名称对于Ollama和aliyun都适用
        "llm_batch_size": 5,
        # 阿里云百炼专属配置 (仅当 LLM_PROVIDER_CHOICE = 'aliyun' 时需要)
        "aliyun_api_key": os.environ.get("ALIYUN_API_KEY"),
        "aliyun_api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1"
    }

    # --- 创建输入文件 ---
    # 使用相对路径，工作目录相对于项目根目录
    input_file_path = WORKING_DIR / "chapter10" / "stru_mech" / "vdb_entities.json"

    # --- 初始化并执行去重流程 ---
    try:
        deduplicator = OptimizedEntityDeduplicator(**config)
        deduplicator.run_deduplication(
            input_file=input_file_path,
            output_prefix="optimized_results"
        )
    except ValueError as e:
        print(f"[配置错误] {e}")
