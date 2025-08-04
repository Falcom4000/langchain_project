import json
import os
from typing import Any

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
import pandas as pd
load_dotenv()

# 初始化 MCP 服务器
mcp = FastMCP("CSVServer")
# OpenAI API 配置
API_KEY = os.getenv("OPENAI_API_KEY")


@mcp.tool()
def csv_mean() -> Any:
    """
    读取 CSV 文件并返回均值信息。
    :return: 包含消息和数据的字典
    """
    df = pd.read_csv("./data/titanic_cleaned.csv")
    mean_values = df.mean().to_dict()
    return {
        "message": "成功获取数值列的均值信息",
        "data": mean_values
    }

@mcp.tool()
def csv_max() -> Any:
    """
    读取 CSV 文件并返回最大值信息。
    :return: 包含消息和数据的字典
    """
    df = pd.read_csv("./data/titanic_cleaned.csv")
    max_values = df.max().to_dict()
    return {
        "message": "成功获取各列的最大值信息",
        "data": max_values
    }

@mcp.tool()
def csv_min() -> Any:
    """
    读取 CSV 文件并返回最小值信息。
    :return: 包含消息和数据的字典
    """
    df = pd.read_csv("./data/titanic_cleaned.csv")
    min_values = df.min().to_dict()
    return {
        "message": "成功获取各列的最小值信息",
        "data": min_values
    }

@mcp.tool()
def csv_info() -> Any:
    """
    读取 CSV 文件并返回数据概览信息。
    :return: 包含消息和数据的字典
    """
    df = pd.read_csv("./data/titanic_cleaned.csv")
    info = {
        "columns": df.columns.tolist(),
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "head": df.head().to_dict(orient='records')
    }
    return {
        "message": "成功获取数据概览信息",
        "data": info
    }

@mcp.tool()
def csv_plot(col_name: str) -> Any:
    """
    读取 CSV 文件并生成数据分布图。
    :param col_name: 列名
    :return: 包含消息和数据的字典
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    df = pd.read_csv("./data/titanic_cleaned.csv")
    
    # 确保图片目录存在
    os.makedirs("./images", exist_ok=True)
    
    # 绘制数据分布图
    plt.figure(figsize=(10, 6))
    sns.histplot(df[col_name], kde=True)
    plt.title(f'Distribution of {col_name}')
    plt.xlabel(col_name)
    plt.ylabel('Frequency')
    image_path = f"./images/{col_name}_distribution.png"
    plt.savefig(image_path)
    plt.close()
    
    return {
        "message": f"成功生成 {col_name} 列的分布图",
        "data": {
            "image_path": image_path,
            "column_name": col_name
        }
    }

@mcp.tool()
def csv_fill_null() -> Any:
    """
    读取 CSV 文件并用均值填充缺失值。
    :return: 包含消息和数据的字典
    """
    df = pd.read_csv("./data/titanic_cleaned.csv")
    
    # 记录原始缺失值情况
    original_nulls = df.isnull().sum().to_dict()
    
    # 只对数值列填充均值
    numeric_columns = df.select_dtypes(include=['number']).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    
    # 保存填充后的数据
    output_path = "./data/titanic_filled.csv"
    df.to_csv(output_path, index=False)
    
    return {
        "message": "成功用均值填充数值列的缺失值",
        "data": {
            "output_file": output_path,
            "filled_columns": numeric_columns.tolist(),
            "original_null_counts": original_nulls
        }
    }



if __name__ == "__main__":
    # 以标准 stdio 方式运行 MCP 服务器
    mcp.run(transport='stdio')