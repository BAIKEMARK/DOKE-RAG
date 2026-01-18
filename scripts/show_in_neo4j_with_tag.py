import os
import json
from dotenv import load_dotenv
from doke_rag.core.utils import xml_to_json
from doke_rag.config.paths import WORKING_DIR
from neo4j import GraphDatabase
import re

# Constants - WORKING_DIR is imported from config module
BATCH_SIZE_NODES = 500
BATCH_SIZE_EDGES = 100

# Neo4j connection credentials

# 显式加载 .env 文件（必须）
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


def convert_xml_to_json(xml_path, output_path):
    """Converts XML file to JSON and saves the output."""
    if not os.path.exists(xml_path):
        print(f"Error: File not found - {xml_path}")
        return None

    json_data = xml_to_json(xml_path)
    if json_data:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"JSON file created: {output_path}")
        return json_data
    else:
        print("Failed to create JSON data")
        return None


def process_in_batches(tx, query, data, batch_size):
    """Process data in batches and execute the given query."""
    for i in range(0, len(data), batch_size):
        batch = data[i: i + batch_size]
        tx.run(query, {"nodes": batch} if "nodes" in query else {"edges": batch})


# 新增函数：清理标签名，确保符合Neo4j标签命名规范
def sanitize_label_name(name):
    # 使用正则表达式替换非法字符为下划线
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    # 确保不以数字或下划线开头
    if sanitized and (sanitized[0].isdigit() or sanitized[0] == '_'):
        sanitized = 'Label_' + sanitized
    return sanitized


def main():
    # Paths
    xml_file = os.path.join(WORKING_DIR, "graph_chunk_entity_relation.graphml")
    json_file = os.path.join(WORKING_DIR, "graph_data.json")

    # Extract label name from working directory
    label_name_raw = os.path.basename(WORKING_DIR)
    # 使用新函数清理标签名
    label_name = sanitize_label_name(label_name_raw)
    # 添加Label_前缀以符合之前的命名习惯（除了清理非法字符）
    label_name = f"Label_translated_{label_name}"
    # label_name = f"Label_merged_manual"

    print(f"Using label name: {label_name}")  # 调试信息

    # Convert XML to JSON
    json_data = convert_xml_to_json(xml_file, json_file)
    if json_data is None:
        return

    # Load nodes and edges
    nodes = json_data.get("nodes", [])
    edges = json_data.get("edges", [])

    # Neo4j queries (修正后的版本)
    create_nodes_query = f"""
    UNWIND $nodes AS node
    MERGE (e:{label_name} {{id: node.id}})
    SET e.entity_type = node.entity_type,
        e.description = node.description,
        e.source_id = node.source_id,
        e.displayName = node.id
    WITH e, node
    // CALL apoc.create.addLabels(e, [node.entity_type]) YIELD node AS n
    RETURN count(*)
    """

    create_edges_query = f"""
    UNWIND $edges AS edge
    MATCH (source:{label_name} {{id: edge.source}})
    MATCH (target:{label_name} {{id: edge.target}})
    WITH source, target, edge, 'RELATED_TO' AS relType
    CALL apoc.create.relationship(source, relType, {{
      weight: edge.weight,
      description: edge.description,
      keywords: edge.keywords,
      source_id: edge.source_id
    }}, target) YIELD rel
    RETURN count(*)
    """

    # Create a Neo4j driver
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    try:
        with driver.session() as session:
            # 插入节点（使用文件标签作为主标签）
            session.execute_write(
                process_in_batches, create_nodes_query, nodes, BATCH_SIZE_NODES
            )

            # 插入关系（限定在同标签节点间创建）
            session.execute_write(
                process_in_batches, create_edges_query, edges, BATCH_SIZE_EDGES
            )

    except Exception as e:
        print(f"Error occurred: {e}")

    finally:
        driver.close()


if __name__ == "__main__":
    main()