import asyncio
import pandas as pd
import os
import logging
from pathlib import Path
from dataclasses import asdict
from doke_rag.core import LightRAG, QueryParam
from doke_rag.core.llm.ollama import ollama_model_complete, ollama_embed
from doke_rag.core.llm.openai import openai_complete_if_cache
from doke_rag.core.base import DocStatus,BaseGraphStorage
from doke_rag.core.utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    encode_string_by_tiktoken,
    setup_logger,
    TokenTracker,
    logger,
)
import json
from doke_rag.core.operate import _merge_nodes_then_upsert,_merge_edges_then_upsert
from doke_rag.core.kg.shared_storage import initialize_pipeline_status
from doke_rag.config.paths import WORKING_DIR, ENV_FILE, ensure_dir
from typing import Any

# ============================================
# Configuration (UPDATE THESE PATHS AS NEEDED)
# ============================================

# Expert graph file to insert
EXPERT_GRAPH_FILE = "data/expert_graphs/人工图_chapter6_en.json"

# Optional: Excel file for batch operations
# file_path = "data/input/split_checked_ds14_txt_zh_new.xlsx"

# ============================================

# Ensure working directory exists
ensure_dir(WORKING_DIR)

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
setup_logger("lightrag", level="INFO")

token_tracker = TokenTracker()
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


class CustomLightRAG(LightRAG):
    async def ainsert_custom_kg(
            self, custom_kg: dict[str, Any], full_doc_id: str = None
    ) -> None:
        update_storage = False
        try:
            # ====================== 1. 插入 chunks 到向量库 ======================
            all_chunks_data: dict[str, dict[str, str]] = {}
            chunk_to_source_map: dict[str, str] = {}
            for chunk_data in custom_kg.get("chunks", []):
                chunk_content = self.clean_text(chunk_data["content"])
                source_id = chunk_data["source_id"]
                tokens = len(
                    encode_string_by_tiktoken(
                        chunk_content, model_name=self.tiktoken_model_name
                    )
                )
                chunk_order_index = chunk_data.get("chunk_order_index", 0)
                chunk_id = compute_mdhash_id(chunk_content, prefix="chunk-")

                chunk_entry = {
                    "content": chunk_content,
                    "source_id": source_id,
                    "tokens": tokens,
                    "chunk_order_index": chunk_order_index,
                    "full_doc_id": full_doc_id if full_doc_id is not None else source_id,
                    "status": DocStatus.PROCESSED,
                }
                all_chunks_data[chunk_id] = chunk_entry
                chunk_to_source_map[source_id] = chunk_id
                update_storage = True

            if all_chunks_data:
                await asyncio.gather(
                    self.chunks_vdb.upsert(all_chunks_data),
                    self.text_chunks.upsert(all_chunks_data),
                )

            # ====================== 2. 插入/合并实体节点 ======================
            all_entities_data: list[dict[str, str]] = []
            global_config = asdict(self)

            for entity_data in custom_kg.get("entities", []):
                entity_name = entity_data["entity_name"]
                entity_type = entity_data.get("entity_type", "UNKNOWN")
                description = entity_data.get("description", "No description provided")
                source_chunk_id = entity_data.get("source_id", "UNKNOWN")
                source_id = chunk_to_source_map.get(source_chunk_id, "UNKNOWN")
                if source_id == "UNKNOWN":
                    logger.warning(
                        f"Entity '{entity_name}' has an UNKNOWN source_id. Please check the source mapping."
                    )

                # 构造当前新节点信息
                new_node = {
                    "entity_type": entity_type,
                    "description": description,
                    "source_id": source_id,
                }
                # 对于每个实体，先调用 _merge_nodes_then_upsert，
                # nodes_data 参数为当前新节点（可扩展为多个节点数据合并）
                node_data = await _merge_nodes_then_upsert(
                    entity_name,
                    [new_node],
                    self.chunk_entity_relation_graph,
                    global_config=global_config,
                )
                all_entities_data.append(node_data)
                update_storage = True

            # ====================== 3. 插入/合并关系边 ======================
            all_relationships_data: list[dict[str, str]] = []
            for relationship_data in custom_kg.get("relationships", []):
                src_id = relationship_data["src_id"]
                tgt_id = relationship_data["tgt_id"]
                description = relationship_data["description"]
                keywords = relationship_data["keywords"]
                weight = relationship_data.get("weight", 1.0)
                source_chunk_id = relationship_data.get("source_id", "UNKNOWN")
                source_id = chunk_to_source_map.get(source_chunk_id, "UNKNOWN")
                if source_id == "UNKNOWN":
                    logger.warning(
                        f"Relationship from '{src_id}' to '{tgt_id}' has an UNKNOWN source_id. Please check the source mapping."
                    )

                # 对于该关系，构造本次边数据
                new_edge = {
                    "weight": weight,
                    "description": description,
                    "keywords": keywords,
                    "source_id": source_id,
                }
                # _merge_edges_then_upsert 接受 edges_data 为 list，如果已有多条关系可以合并
                edge_data = await _merge_edges_then_upsert(
                    src_id,
                    tgt_id,
                    [new_edge],
                    self.chunk_entity_relation_graph,
                    global_config=global_config,
                )
                all_relationships_data.append(edge_data)
                update_storage = True

            # ====================== 4. 更新实体向量库 ======================
            data_for_vdb = {
                compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                    "content": dp["entity_name"] + "\n" + dp["description"],
                    "entity_name": dp["entity_name"],
                    "source_id": dp["source_id"],
                    "description": dp["description"],
                    "entity_type": dp["entity_type"],
                }
                for dp in all_entities_data
            }
            await self.entities_vdb.upsert(data_for_vdb)

            # ====================== 5. 更新关系向量库 ======================
            data_for_vdb = {
                compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                    "src_id": dp["src_id"],
                    "tgt_id": dp["tgt_id"],
                    "source_id": dp["source_id"],
                    "content": f"{dp['keywords']}\t{dp['src_id']}\n{dp['tgt_id']}\n{dp['description']}",
                    "keywords": dp["keywords"],
                    "description": dp["description"],
                    "weight": dp.get("weight", 1.0),
                }
                for dp in all_relationships_data
            }
            await self.relationships_vdb.upsert(data_for_vdb)

        except Exception as e:
            logger.error(f"Error in ainsert_custom_kg: {e}")
            raise
        finally:
            if update_storage:
                await self._insert_done()


async def llm_model_func_aliyun(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        "deepseek-r1-0528",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("ALIYUN_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里云
        token_tracker=token_tracker,
        **kwargs,
    )

# 初始化
async def initialize_rag():
    rag = CustomLightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func_aliyun,
        llm_model_max_async=4,
        llm_model_max_token_size=32768,
        embedding_func=EmbeddingFunc(
            embedding_dim=768,  # 768 for nomic-embed-text:latest
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts, embed_model="nomic-embed-text:latest", host="http://localhost:11434"
            ),
        ),
        entity_extract_max_gleaning=1,
        addon_params={
            "example_number": 4,  # 设置为3即可启用3个示例
            "insert_batch_size": 50,  # Process 20 documents per batch
            "entity_types": ["method", "concept", "equation", "structure", "constraint", "process", "step"],
            "language": "简体中文",
        },
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag

# 将主逻辑包装在异步函数中
async def delete_false_node(rag):
    rag = await initialize_rag()
    token_tracker.reset()

    # 载入 Excel 文件
    df_nodes = pd.read_excel(file_path, sheet_name="Nodes")

    # 过滤 assessment == "F" 的行
    f_entities = df_nodes[df_nodes["assessment"] == "F"]

    # 获取对应的 id（作为实体名）
    ids_to_delete = f_entities["id"].tolist()

    # 打印检查
    print("需要删除的实体ID数量：", len(ids_to_delete))

    # 删除操作（循环执行）
    for entity_name in ids_to_delete:
        print(f"删除实体：{entity_name}")
        await rag.adelete_by_entity(entity_name)

    print("Token usage:", token_tracker.get_usage())

async def _insert_custom_kg(rag):
    # 读取专家图 JSON 文件
    expert_graph_path = Path(EXPERT_GRAPH_FILE)

    if not expert_graph_path.exists():
        logger.warning(f"Expert graph file not found: {expert_graph_path}")
        logger.info("Skipping expert graph insertion.")
        return

    with open(expert_graph_path, "r", encoding="utf-8") as file:
        custom_kg = json.load(file)

    await rag.ainsert_custom_kg(custom_kg)
    logger.info(f"Expert graph inserted successfully from: {expert_graph_path}")
async def _edit_entity(rag):
     await rag.aedit_entity(
        "力法方程",
        {"description": r"""力法方程是结构力学中力法分析的核心工具，主要用于求解超静定结构的多余未知力。其核心原理是通过变形协调条件，确保所选基本体系（或基本结构）的受力变形与原结构完全等价。具体而言，方程以多余约束力（即未知赘余力 $X_i$）为未知量，建立力与位移之间的关系，从而消除基本体系与原结构之间的位移差异。  

对于单次超静定结构，力法方程的标准形式为：  
$$  
\delta_{11}X_1 + \Delta_{1P} = 0  
$$  
其中，$\delta_{11}$ 表示单位力 $X_1 = 1$ 作用下在方向 $X_1$ 产生的位移，$\Delta_{1P}$ 表示外荷载作用下在同一方向产生的位移。该方程确保基本体系与原结构在赘余约束处的位移协调性（即位移为零）。  

对于 $n$ 次超静定结构，方程扩展为矩阵形式：  
$$  
\begin{bmatrix}  
\delta_{11} & \delta_{12} & \cdots & \delta_{1n} \\  
\delta_{21} & \delta_{22} & \cdots & \delta_{2n} \\  
\vdots & \vdots & & \vdots \\  
\delta_{n1} & \delta_{n2} & \cdots & \delta_{nn}  
\end{bmatrix}  
\begin{Bmatrix}  
X_{1} \\  
X_{2} \\  
\vdots \\  
X_{n}  
\end{Bmatrix}  
+  
\begin{Bmatrix}  
\Delta_{1P} \\  
\Delta_{2P} \\  
\vdots \\  
\Delta_{nP}  
\end{Bmatrix}  
=  
\begin{Bmatrix}  
0 \\  
0 \\  
\vdots \\  
0  
\end{Bmatrix}  
$$  
式中，$\delta_{ij}$ 表示单位力 $X_j = 1$ 在方向 $X_i$ 产生的位移，$\Delta_{iP}$ 为外荷载在方向 $X_i$ 产生的位移，方程右侧为零向量，表明所有赘余约束处的位移必须与原结构一致（即无实际位移）。  

通过上述方程，可系统求解超静定结构的未知力，并保证受力与变形的等效性。关于力法方程的详细图示可参考：![](https://rag-img.051088.xyz/Force_Method/力法方程详解.png)。"""})


async def merge_duplicate_entities(rag):
    """
    读取 merge_groups_with_llm.json 文件，并依次调用 rag.merge_entities 命令，
    将每个组中的待合并实体（to_merge）与代表实体（representative）进行合并，
    合并策略为：description 和 entity_type 使用 "concatenate"，source_id 使用 "join_unique"。
    """
    # 读取 JSON 文件
    with open("dedup_outputs/optimized_results_merged.json", "r", encoding="utf-8") as f:
        merge_groups = json.load(f)
    # 遍历每一组数据
    for group in merge_groups:
        representative = group["representative"]
        to_merge = group["to_merge"]

        # 调用合并命令，执行实体合并
        await  rag.amerge_entities(
            source_entities=to_merge,
            target_entity=representative,
            merge_strategy={
                "description": "concatenate",
                "entity_type": "join_unique",
                "source_id": "join_unique"
            }
        )

async def main():
    rag = await initialize_rag()
    token_tracker.reset()
    # asyncio.run(delete_false_node())
    await _insert_custom_kg(rag)
    # await _edit_entity(rag)
    # await merge_duplicate_entities(rag)
    print("Token usage:", token_tracker.get_usage())



# 运行主逻辑
if __name__ == "__main__":
    asyncio.run(main())
