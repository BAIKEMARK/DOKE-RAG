import os
import networkx as nx
import pandas as pd
from pathlib import Path
from doke_rag.config.paths import WORKING_DIR

def process_graphml_folder(folder_path, output_dir):
    """
    处理指定文件夹下的 .graphml 文件，并将数据转换为 Excel。
    :param folder_path: str, 指定的 GraphML 文件所在文件夹
    :param output_dir: str, Excel 输出目录
    """
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print(f"❌ 错误: 指定的文件夹 '{folder_path}' 不存在或不是文件夹。")
        return

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 查找 .graphml 文件
    graphml_files = [f for f in os.listdir(folder_path) if f.endswith(".graphml")]
    if not graphml_files:
        print(f"⚠️ 警告: 在 '{folder_path}' 中未找到 .graphml 文件。")
        return

    for file_name in graphml_files:
        file_path = os.path.join(folder_path, file_name)
        output_file_path = os.path.join(output_dir, f"{os.path.basename(folder_path)}.xlsx")

        try:
            # 读取 GraphML 文件
            G = nx.read_graphml(file_path)

            # 处理节点数据
            df_nodes = pd.DataFrame([
                {"id": node_id, "entity_type": attrs.get("entity_type", ""),
                 "description": attrs.get("description", "")}
                for node_id, attrs in G.nodes(data=True)
            ])

            # 处理边数据
            df_edges = pd.DataFrame([
                {"source": source, "target": target,
                 "weight": attrs.get("weight", 0),
                 "description": attrs.get("description", ""),
                 "keywords": attrs.get("keywords", "")}
                for source, target, attrs in G.edges(data=True)
            ])

            # 处理孤立节点（度数为 0）
            df_isolated_nodes = df_nodes[df_nodes["id"].isin([n for n in G.nodes() if G.degree(n) == 0])]

            # 写入 Excel
            with pd.ExcelWriter(output_file_path) as writer:
                df_nodes.to_excel(writer, sheet_name="Nodes", index=False)
                df_edges.to_excel(writer, sheet_name="Edges", index=False)
                df_isolated_nodes.to_excel(writer, sheet_name="Isolated_Nodes", index=False)

            print(f"✅ 处理完成: {file_path} -> {output_file_path}")

        except Exception as e:
            print(f"❌ 处理失败: {file_path}, 错误: {e}")

    print(f"🎉 处理完成: {folder_path}\n")


def process_run_data(root_dir, folder_name=None):
    """
    处理 `run_data` 目录下所有子文件夹，或仅处理指定文件夹
    :param root_dir: str, `run_data` 目录路径
    :param folder_name: str, 指定的文件夹名称（可选）
    """
    output_dir = os.path.join(root_dir, "EXCEL/cache")

    # 如果指定了 `folder_name`，只处理该文件夹
    if folder_name:
        folder_path = os.path.join(root_dir, folder_name)
        process_graphml_folder(folder_path, output_dir)
    else:
        # 遍历所有子文件夹
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                process_graphml_folder(folder_path, output_dir)


# 示例: 运行代码
if __name__ == "__main__":
    # 使用相对路径，工作目录相对于项目根目录
    root_dir = WORKING_DIR

    # ✅ 处理单个文件夹
    # 请根据实际需要修改文件夹名称
    process_run_data(root_dir, folder_name="unsplited_0811")

    # ✅ 处理 `run_data` 下的所有子文件夹
    # process_run_data(root_dir)  # 取消注释此行即可处理所有文件夹
