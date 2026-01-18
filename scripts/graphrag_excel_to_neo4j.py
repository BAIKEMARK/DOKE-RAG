
# Excel—>neo4j
def excel_to_neo4j():
    import os
    import pandas as pd
    from pathlib import Path
    from dotenv import load_dotenv
    from neo4j import GraphDatabase
    from doke_rag.config.paths import WORKING_DIR

    # Load environment variables
    load_dotenv()

    # ======== 配置区域 ========
    NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

    # 可设为某个 Excel 文件或文件夹路径 - 使用相对路径
    TARGET_PATH = WORKING_DIR / "EXCEL"
    # ==========================

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def get_excel_files(path):
        if path.lower().endswith(".xlsx"):
            return [path]  # 单文件处理
        else:
            return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".xlsx")]

    def create_or_merge_node(tx, node_id, description, assessment, base_label):
        labels = [base_label]

        if pd.isna(assessment) or str(assessment).strip().lower() == "none":
            labels.append("空节点")
        else:
            labels.append(str(assessment))

        label_str = ":".join(labels)

        query = f"""
        MERGE (n:{label_str} {{id: $node_id}})
        ON CREATE SET n.description = $description
        ON MATCH SET n.description =
            CASE
                WHEN n.description CONTAINS $description THEN n.description
                ELSE n.description + '<合并>' + $description
            END
        """
        tx.run(query, node_id=node_id, description=description)

    def create_relationship(tx, source_id, target_id, rel_type, description, base_label):
        if not pd.notna(rel_type):
            rel_type = "RELATED_TO"
        if not pd.notna(description):
            description = ""

        # 限定 base_label 保证仅连接当前图谱内节点
        query = f'''
        MATCH (a:`{base_label}` {{id: $source_id}})
        MATCH (b:`{base_label}` {{id: $target_id}})
        MERGE (a)-[r:{rel_type} {{description: $description}}]->(b)
        '''
        tx.run(query, source_id=source_id, target_id=target_id, description=description)

    def process_excel(file_path):
        print(f"\n📄 处理文件: {file_path}")
        base_label = os.path.splitext(os.path.basename(file_path))[0]

        try:
            nodes_df = pd.read_excel(file_path, sheet_name="Nodes")
            edges_df = pd.read_excel(file_path, sheet_name="Edges")
        except Exception as e:
            print(f"⚠️  读取失败: {e}")
            return

        with driver.session() as session:
            for _, row in nodes_df.iterrows():
                session.write_transaction(
                    create_or_merge_node,
                    node_id=row["id"],
                    description=row.get("description", ""),
                    assessment=row.get("assessment", None),
                    base_label=base_label,
                )

            for _, row in edges_df.iterrows():
                session.write_transaction(
                    create_relationship,
                    source_id=row["source"],
                    target_id=row["target"],
                    rel_type=row.get("type", "RELATED_TO"),
                    description=row.get("description", ""),
                    base_label=base_label
                )

        print(f"✅ 完成导入: {base_label}")

    # ========== 执行入口 ==========
    excel_files = get_excel_files(TARGET_PATH)

    for file in excel_files:
        process_excel(file)

    driver.close()
    print("\n🎉 所有图谱已导入完成 ✅")


# neo4j脚本批量生成
def neo4j_cypher_generate():
    from pathlib import Path

    # 所有标签名及其文件名
    labels = [
        "Graphrag_raw_checked",
        "Graphrag_split_checked_txt",
        "Lightrag_raw",
        "Lightrag_raw_checked",
        "Lightrag_raw_checked_zh",
        "Lightrag_raw_checked_zh_ds671b",
        "Lightrag_split",
        "Lightrag_split_checked_json",
        "Lightrag_split_checked_json_zh",
        "Lightrag_split_checked_json_zh_ds671b",
        "Lightrag_split_checked_txt",
        "Lightrag_split_checked_txt_zh",
    ]

    output_dir = Path("D:/Desktop/cypher_exports")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = []

    for label in labels:
        filename = output_dir / f"{label}只看T.cypher"
        content = f"""// {label}_T
    MATCH (n:{label})
    WHERE 'T' IN labels(n) AND NOT '空节点' IN labels(n) AND NOT 'F' IN labels(n)
    RETURN n
    LIMIT 300
    """
        filename.write_text(content, encoding='utf-8')
        files.append(filename)

    print("✅ 所有查询含 T 标签的 .cypher 文件已生成！")

if __name__ == "__main__":
    excel_to_neo4j()
    print('end')
    pass