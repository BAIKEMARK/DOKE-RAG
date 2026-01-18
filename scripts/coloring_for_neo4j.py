import pandas as pd
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError
import os
from pathlib import Path
from dotenv import load_dotenv
from doke_rag.config.paths import WORKING_DIR

# 显式加载 .env 文件
load_dotenv()

# 配置信息
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


def get_file_label(file_path):
    """从文件路径中提取文件名（不含扩展名）作为标签"""
    return os.path.splitext(os.path.basename(file_path))[0]


def read_excel_data(file_path):
    """读取Excel中的节点和边数据"""
    try:
        # 读取节点数据
        nodes_df = pd.read_excel(
            file_path,
            sheet_name="Nodes",
            usecols=["id", "assessment"],
            dtype={"assessment": "category"}
        ).dropna(subset=["id"])

        # 读取边数据
        edges_df = pd.read_excel(
            file_path,
            sheet_name="Edges",
            usecols=["source", "target", "assessment"],
            dtype={"assessment": "category"}
        ).dropna(subset=["source", "target"])

        return nodes_df, edges_df

    except Exception as e:
        print(f"Excel读取失败: {str(e)}")
        raise


def update_neo4j_labels(file_path, nodes_df, edges_df):
    """更新Neo4j中的节点标签和边属性"""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    file_label = get_file_label(file_path)  # 获取文件名作为标签

    try:
        with driver.session() as session:
            # 清除旧标签和属性（可选）
            session.run(f"MATCH (n:{file_label}) REMOVE n:NS:T:F RETURN count(n)")
            session.run(f"MATCH (:{file_label})-[r]->(:{file_label}) REMOVE r.assessment RETURN count(r)")

            # 准备节点批量数据
            nodes_batch = [{
                "id": row["id"],
                "label": row["assessment"] if pd.notna(row["assessment"]) else "NS"
            } for _, row in nodes_df.iterrows()]

            # 节点更新查询（使用APOC动态添加标签）
            node_update_query = f"""
            UNWIND $batch AS row
            MATCH (n:{file_label} {{id: row.id}})
            CALL apoc.create.addLabels(n, [row.label]) YIELD node
            RETURN node.id
            """
            nodes_result = session.run(node_update_query, {"batch": nodes_batch})
            print(f"节点标签更新记录：{len(list(nodes_result))}")

            # 准备边批量数据
            edges_batch = [{
                "source": row["source"],
                "target": row["target"],
                "assessment": row["assessment"] if pd.notna(row["assessment"]) else "NS"
            } for _, row in edges_df.iterrows()]

            # 边更新查询
            edges_update_query = f"""
            UNWIND $batch AS row
            MATCH (s:{file_label} {{id: row.source}})-[r]->(t:{file_label} {{id: row.target}})
            SET r.assessment = row.assessment  // 为边添加属性
            RETURN type(r)
            """
            edges_result = session.run(edges_update_query, {"batch": edges_batch})
            print(f"边属性更新记录：{len(list(edges_result))}")

    except Neo4jError as e:
        print(f"Neo4j操作失败: {e.code}: {e.message}")
    finally:
        driver.close()
# def update_neo4j_labels(file_path, nodes_df, edges_df):
#     """更新Neo4j中的节点标签和边属性"""
#     driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
#     file_label = get_file_label(file_path)  # 获取文件名作为标签
#
#     try:
#         with driver.session() as session:
#             # 清除旧标签和属性（可选）
#             session.run(f"MATCH (n:{file_label}) REMOVE n:NS:T:F RETURN count(n)")
#             session.run(f"MATCH (:{file_label})-[r]->(:{file_label}) REMOVE r.assessment RETURN count(r)")
#
#             # 准备节点批量数据
#             nodes_batch = [{
#                 "id": row["id"],
#                 "label": row["assessment"] if pd.notna(row["assessment"]) else "NS",
#                 "color": {
#                     "T": "#4CAF50",  # 绿色
#                     "F": "#F44336",  # 红色
#                     "NS": "#9E9E9E"  # 灰色
#                 }.get(row["assessment"], "#9E9E9E")  # 默认灰色
#             } for _, row in nodes_df.iterrows()]
#
#             # 节点更新查询（添加颜色属性）
#             node_update_query = f"""
#             UNWIND $batch AS row
#             MATCH (n:{file_label} {{id: row.id}})
#             SET n.color = row.color  // 添加颜色属性
#             CALL apoc.create.addLabels(n, [row.label]) YIELD node
#             RETURN node.id
#             """
#             nodes_result = session.run(node_update_query, {"batch": nodes_batch})
#             print(f"节点标签更新记录：{len(list(nodes_result))}")
#
#             # 准备边批量数据
#             edges_batch = [{
#                 "source": row["source"],
#                 "target": row["target"],
#                 "assessment": row["assessment"] if pd.notna(row["assessment"]) else "NS"
#             } for _, row in edges_df.iterrows()]
#
#             # 边更新查询
#             edges_update_query = f"""
#             UNWIND $batch AS row
#             MATCH (s:{file_label} {{id: row.source}})-[r]->(t:{file_label} {{id: row.target}})
#             SET r.assessment = row.assessment  // 为边添加属性
#             RETURN type(r)
#             """
#             edges_result = session.run(edges_update_query, {"batch": edges_batch})
#             print(f"边属性更新记录：{len(list(edges_result))}")
#
#     except Neo4jError as e:
#         print(f"Neo4j操作失败: {e.code}: {e.message}")
#     finally:
#         driver.close()

def process_file(file_path):
    """处理单个文件"""
    try:
        print(f"\n处理文件: {file_path}")
        nodes_df, edges_df = read_excel_data(file_path)

        # 打印数据摘要
        print(f"读取到 {len(nodes_df)} 个节点记录")
        print(f"读取到 {len(edges_df)} 条边记录")
        print("节点assessment分布：\n", nodes_df["assessment"].value_counts())
        print("边assessment分布：\n", edges_df["assessment"].value_counts())

        # 更新数据库
        update_neo4j_labels(file_path, nodes_df, edges_df)

    except Exception as e:
        print(f"文件处理失败: {str(e)}")


def process_folder(folder_path):
    """处理文件夹下的所有Excel文件"""
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".xlsx"):
            file_path = os.path.join(folder_path, file_name)
            process_file(file_path)


def main():
    # 选择处理单个文件或文件夹 - 使用相对路径
    # 请根据实际需要修改文件名
    target_path = WORKING_DIR / "EXCEL" / "qwen_split_数据库改进_zh_txt_ds14b.xlsx"
    if os.path.isdir(target_path):
        print(f"处理文件夹: {target_path}")
        process_folder(target_path)
    elif os.path.isfile(target_path) and target_path.endswith(".xlsx"):
        process_file(target_path)
    else:
        print("路径无效，请提供有效的Excel文件或文件夹路径。")


if __name__ == "__main__":
    main()