import pandas as pd
from pathlib import Path
from doke_rag.config.paths import RESULTS_DIR, EXCEL_RESULTS

# --- 1. 配置 ---
# 请将这里设置为您要扫描的根目录
SEARCH_ROOT_DIRECTORY = RESULTS_DIR / "experiment_evaluation"

# 您想要查找的CSV文件名
TARGET_CSV_FILENAME = "final_statistical_summary.csv"

# 最终生成的Excel文件名 - 从配置模块导入
OUTPUT_EXCEL_FILE = EXCEL_RESULTS
# --- 结束配置 ---


def find_csv_files(root_dir: Path, filename: str) -> list:
    """在根目录下递归查找所有指定名称的CSV文件"""
    print(f"正在在 '{root_dir}' 中搜索 '{filename}'...")
    files = list(root_dir.rglob(filename))
    if not files:
        print(
            "未找到任何匹配的CSV文件。请检查 SEARCH_ROOT_DIRECTORY 和 TARGET_CSV_FILENAME 是否正确。"
        )
    else:
        print(f"成功找到 {len(files)} 个文件。")
    return files


def process_and_write_to_excel(csv_files: list, output_path: str):
    """
    处理所有找到的CSV文件，并将它们格式化后写入单个Excel工作表。
    """
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        start_row = 0

        # 定义我们想要的列的顺序
        dims_order = ["Comprehensiveness", "Diversity", "Empowerment", "Overall"]

        for i, file_path in enumerate(csv_files):
            try:
                experiment_name = file_path.parent.name
                print(f"({i + 1}/{len(csv_files)}) 正在处理: {experiment_name}")

                df = pd.read_csv(file_path)

                # ==================== 优化点 ====================
                # 不再硬编码行顺序，而是从文件中动态获取
                # .unique() 会按首次出现的顺序返回唯一值
                row_order = df["challenger_group"].unique()
                # ===============================================

                # --- 创建并排序平均值表格 ---
                mean_table = df.pivot(
                    index="challenger_group", columns="metric", values="mean"
                )
                mean_table = mean_table.reindex(columns=dims_order)
                # 使用从文件中学习到的顺序对行进行排序
                mean_table = mean_table.reindex(row_order)

                # --- 创建并排序标准差表格 ---
                std_dev_table = df.pivot(
                    index="challenger_group", columns="metric", values="std_dev"
                )
                std_dev_table = std_dev_table.reindex(columns=dims_order)
                # 使用从文件中学习到的顺序对行进行排序
                std_dev_table = std_dev_table.reindex(row_order)

                # --- 写入Excel ---
                header_df = pd.DataFrame([f"实验组: {experiment_name}"])
                header_df.to_excel(
                    writer,
                    sheet_name="Summary",
                    startrow=start_row,
                    index=False,
                    header=False,
                )
                start_row += 2

                mean_header_df = pd.DataFrame(["平均值 (Mean Win Rate)"])
                mean_header_df.to_excel(
                    writer,
                    sheet_name="Summary",
                    startrow=start_row,
                    index=False,
                    header=False,
                )
                start_row += 1
                mean_table.to_excel(writer, sheet_name="Summary", startrow=start_row)
                start_row += len(mean_table.index) + 2

                std_dev_header_df = pd.DataFrame(["标准差 (Standard Deviation)"])
                std_dev_header_df.to_excel(
                    writer,
                    sheet_name="Summary",
                    startrow=start_row,
                    index=False,
                    header=False,
                )
                start_row += 1
                std_dev_table.to_excel(writer, sheet_name="Summary", startrow=start_row)
                start_row += len(std_dev_table.index) + 3

            except FileNotFoundError:
                print(f"  [错误] 文件不存在: {file_path}")
            except Exception as e:
                print(f"  [错误] 处理文件 {file_path} 时发生错误: {e}")

    print("\n处理完成！")
    print(f"所有结果已汇总到: {Path.cwd() / output_path}")


def main():
    """主执行函数"""
    csv_files_to_process = find_csv_files(SEARCH_ROOT_DIRECTORY, TARGET_CSV_FILENAME)
    if csv_files_to_process:
        process_and_write_to_excel(csv_files_to_process, OUTPUT_EXCEL_FILE)

if __name__ == "__main__":
    # 确保已安装所需库: pip install pandas openpyxl
    main()