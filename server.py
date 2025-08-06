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

@mcp.tool()
def train_survival_model() -> Any:
    """
    训练预测Titanic乘客生存情况的机器学习模型。
    使用70%训练集，10%验证集，20%测试集进行数据切分。
    :return: 包含模型性能指标的字典
    """
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import joblib
    import os
    
    # 读取数据
    df = pd.read_csv("./data/titanic_filled.csv")
    
    # 数据预处理
    # 选择特征列（去除非预测特征）
    feature_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    
    # 处理缺失值
    df_model = df[feature_columns + ['Survived']].copy()
    df_model = df_model.dropna()
    
    # 对分类变量进行编码
    label_encoders = {}
    categorical_columns = ['Sex', 'Embarked']
    
    for col in categorical_columns:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].astype(str))
        label_encoders[col] = le
    
    # 分离特征和目标变量
    X = df_model[feature_columns]
    y = df_model['Survived']
    
    # 第一次分割：分离出测试集（20%）
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 第二次分割：从剩余80%中分离出验证集（10%/80% = 12.5%）
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp
    )
    
    # 训练随机森林模型
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    model.fit(X_train, y_train)
    
    # 预测和评估
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    
    # 计算准确率
    train_accuracy = accuracy_score(y_train, train_pred)
    val_accuracy = accuracy_score(y_val, val_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    # 特征重要性
    feature_importance = dict(zip(feature_columns, model.feature_importances_))
    
    # 保存模型和编码器
    os.makedirs("./models", exist_ok=True)
    joblib.dump(model, "./models/survival_model.pkl")
    joblib.dump(label_encoders, "./models/label_encoders.pkl")
    
    # 生成详细报告
    test_report = classification_report(y_test, test_pred, output_dict=True)
    confusion_mat = confusion_matrix(y_test, test_pred).tolist()
    
    return {
        "message": "成功训练生存预测模型",
        "data": {
            "dataset_info": {
                "total_samples": len(df_model),
                "train_size": len(X_train),
                "validation_size": len(X_val),
                "test_size": len(X_test),
                "features": feature_columns
            },
            "model_performance": {
                "train_accuracy": round(train_accuracy, 4),
                "validation_accuracy": round(val_accuracy, 4),
                "test_accuracy": round(test_accuracy, 4)
            },
            "feature_importance": {k: round(v, 4) for k, v in feature_importance.items()},
            "test_classification_report": test_report,
            "confusion_matrix": confusion_mat,
            "model_saved_path": "./models/survival_model.pkl",
            "encoders_saved_path": "./models/label_encoders.pkl"
        }
    }



if __name__ == "__main__":
    # 以标准 stdio 方式运行 MCP 服务器
    mcp.run(transport='stdio')