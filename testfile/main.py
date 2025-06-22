# main.py (修正版 - 尊重您的路径)

import json
import numpy as np
import sys
import os

# --- 说明 ---
# 本脚本为“AI硬件平台功耗模型”任务的最终执行文件。
# 它不依赖任何第三方库（除了numpy），符合竞赛要求。
# 脚本会加载 `Model/power_model.npz` 文件，对测试数据进行预测，
# 并将结果输出到 `results.jsonl` 文件。
# ----------------

def relu(x):
    """ReLU 激活函数"""
    return np.maximum(0, x)

def predict_from_params(features, params):
    """
    【必须与训练脚本匹配】使用numpy从保存的参数进行预测。
    
    Args:
        features (np.array): 标准化后的1D输入特征向量。
        params (dict): 包含权重和偏置的字典。
        
    Returns:
        float: 预测的功耗值。
    """
    # 这是一个拥有3个隐藏层的MLP的前向传播
    # 这个结构必须和生成 .npz 文件的 train.py 中的结构完全一致
    h1 = relu(np.dot(features, params['w1']) + params['b1'])
    h2 = relu(np.dot(h1, params['w2']) + params['b2'])
    h3 = relu(np.dot(h2, params['w3']) + params['b3'])
    output = np.dot(h3, params['w_out']) + params['b_out']
    return float(output)


def run_prediction():
    """
    执行完整的预测流程：加载数据 -> 加载模型 -> 预测 -> 保存结果。
    """
    # --- 【修改点 1】保留您指定的模型路径 ---
    # 我们将使用您指定的这个路径。
    # 请务必确认，这个文件是由一个定义了3个隐藏层的train.py生成的。
    model_path = 'Model/Model/power_model_deep_simple.npz'
    
    # --- 【修改点 2】修复测试数据文件路径的语法警告 ---
    # 使用 r'' 来创建原始字符串，这是处理Windows路径的最佳方式
    test_data_path = r'E:\大三夏季学期\比赛\线上阶段数据集\data_test.jsonl'
    
    output_path = 'results.jsonl'
    
    # 加载模型参数
    if not os.path.exists(model_path):
        sys.stderr.write(f"错误: 模型文件 '{model_path}' 未找到。\n")
        sys.stderr.write("请检查路径是否正确，以及是否已成功运行train.py生成了模型文件。\n")
        sys.exit(1)
    
    print(f"从 {model_path} 加载模型参数...")
    params = np.load(model_path)
    
    # 检查模型文件内容是否匹配（可选但推荐）
    if 'w3' not in params:
        sys.stderr.write(f"错误: 模型文件 '{model_path}' 中没有找到 'w3'。\n")
        sys.stderr.write("这说明该模型不是由3个隐藏层的网络生成的。请用正确的train.py重新生成模型文件。\n")
        sys.exit(1)
        
    # 加载标准化器参数
    scaler_mean = params['scaler_mean']
    scaler_scale = params['scaler_scale']

    results_to_write = []
    
    # 加载并处理测试数据
    if not os.path.exists(test_data_path):
        sys.stderr.write(f"错误: 测试数据文件 '{test_data_path}' 未找到。\n")
        sys.exit(1)
        
    print(f"从 {test_data_path} 加载测试数据...")
    with open(test_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            feature_vector = np.array(data['Feature'], dtype=np.float32).flatten().reshape(1, -1)
            
            # 应用与训练时相同的标准化
            feature_vector_scaled = (feature_vector - scaler_mean) / scaler_scale
            
            # 进行预测
            prediction = predict_from_params(feature_vector_scaled.flatten(), params)
            results_to_write.append({"PredictResult": prediction})

    # 将结果写入文件
    with open(output_path, 'w') as f:
        for result in results_to_write:
            f.write(json.dumps(result) + '\n')
            
    print(f"预测完成，结果已保存至 {output_path}")


if __name__ == '__main__':
    run_prediction()