# Model/train.py (深度网络 + 简单验证版)

import json
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
import sys

# --- 说明 ---
# 这是一个简洁的优化版本，核心是：
# 1. 使用简单、快速的“训练集-验证集”划分。
# 2. 专注于使用更深、更优的网络结构来提升模型性能。
# 3. 提供清晰的训练集/验证集RMSE对比，以判断过拟合。
# ----------------

def load_and_preprocess_data(file_path):
    """
    加载并预处理 data_train.jsonl 文件。
    对3次采样的特征和标签取平均值。
    """
    print(f"从 {file_path} 加载数据...")
    features_list = []
    labels_list = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # 对特征和标签进行平均
            avg_features = np.mean(data['Features'], axis=0)
            avg_label = np.mean(data['Labels'])
            features_list.append(avg_features)
            labels_list.append(avg_label)

    return np.array(features_list, dtype=np.float32), np.array(labels_list, dtype=np.float32)


def train_deep_network_simple():
    """
    在一个简单的训练/验证集上，训练一个更深的网络模型。
    """
    train_file = 'E:\大三夏季学期\比赛\线上阶段数据集\data_train.jsonl'
    if not os.path.exists(train_file):
        print(f"错误: 训练文件 '{train_file}' 未找到。请将训练数据放置在项目根目录。", file=sys.stderr)
        return

    features, labels = load_and_preprocess_data(train_file)
    print(f"数据加载完成，特征维度: {features.shape}, 标签维度: {labels.shape}")

    # 数据标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 划分训练集和验证集 (80% 训练, 20% 验证)
    X_train, X_val, y_train, y_val = train_test_split(
        features_scaled, labels, test_size=0.2, random_state=42
    )
    print(f"数据集划分完成。训练集: {X_train.shape[0]}条, 验证集: {X_val.shape[0]}条")

    print("\n开始训练深度 MLP 回归模型...")
    # 【优化点: 网络结构与超参数】
    # 您可以专注于调整这里的参数来优化模型
    mlp = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64), # 3个隐藏层的深度网络
        activation='relu',
        solver='adam',
        alpha=0.0005,  # L2 正则化强度
        batch_size=32,
        learning_rate_init=0.001,
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=20,
        verbose=True # 打印每个周期的损失，方便观察
    )

    mlp.fit(X_train, y_train)
    print("\n模型训练完成。")

    # --- 在训练集和验证集上评估模型 ---
    y_train_pred = mlp.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    
    y_val_pred = mlp.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    
    print("\n--- 模型评估结果 ---")
    print(f"训练集 RMSE: {train_rmse:.4f}")
    print(f"验证集 RMSE: {val_rmse:.4f}")
    print("--------------------")
    if train_rmse < val_rmse:
        print("提示: 验证集RMSE高于训练集，模型可能存在一定程度的过拟合。可尝试增大alpha值。")

    # --- 关键步骤: 提取参数并保存 ---
    if not os.path.exists('Model'):
        os.makedirs('Model')

    if len(mlp.coefs_) != 4 or len(mlp.intercepts_) != 4:
       raise ValueError("模型结构与预期的3个隐藏层不符，请检查MLP定义。")

    model_params = {
        'w1': mlp.coefs_[0], 'b1': mlp.intercepts_[0],
        'w2': mlp.coefs_[1], 'b2': mlp.intercepts_[1],
        'w3': mlp.coefs_[2], 'b3': mlp.intercepts_[2],
        'w_out': mlp.coefs_[3], 'b_out': mlp.intercepts_[3],
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_
    }

    save_path = 'Model/power_model_deep_simple.npz'
    np.savez(save_path, **model_params)
    print(f"\n模型参数已成功保存至 {save_path}")
    print("\n重要提示：请确保您的 main.py 推理代码已更新，以匹配新的3个隐藏层结构！")


if __name__ == '__main__':
    train_deep_network_simple()