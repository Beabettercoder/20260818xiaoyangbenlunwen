#!/usr/bin/env python3
"""
快速验证 Test-time Text Calibration 的核心逻辑
"""
import torch
import torch.nn.functional as F
import numpy as np

def test_calibration_logic():
    print("=" * 60)
    print("测试: Text Calibration 核心逻辑")
    print("=" * 60)
    
    # 模拟参数
    n_way = 5
    n_query = 15
    text_calibration_weight = 0.2
    
    # 模拟 GNN 输出的分数
    np.random.seed(42)
    scores = torch.randn(n_way * n_query, n_way)
    print(f"原始 GNN scores shape: {scores.shape}")
    print(f"原始 scores[0]: {scores[0].numpy()}")
    
    # 模拟文本嵌入 (来自 CLIP)
    # 假设5个类的文本嵌入
    text_emb = torch.randn(n_way, 512)
    text_emb_norm = F.normalize(text_emb, dim=-1)
    
    # 计算类间相似度
    text_sim_matrix = torch.mm(text_emb_norm, text_emb_norm.t())
    print(f"\n文本相似度矩阵:\n{text_sim_matrix.numpy()}")
    
    # 计算区分度
    mask = 1.0 - torch.eye(n_way)
    inter_class_sim = (text_sim_matrix * mask).sum(dim=1) / (n_way - 1)
    text_distinctiveness = 1.0 - inter_class_sim
    print(f"\n类别区分度: {text_distinctiveness.numpy()}")
    
    # 转换为 boost 权重
    text_boost = F.softmax(text_distinctiveness * 5.0, dim=0)
    text_boost = text_boost / text_boost.mean()
    print(f"Boost 权重: {text_boost.numpy()}")
    
    # 应用校准
    gnn_probs = F.softmax(scores, dim=-1)
    text_boost_expanded = text_boost.unsqueeze(0)
    
    w = text_calibration_weight
    calibrated_probs = (1 - w) * gnn_probs + w * (gnn_probs * text_boost_expanded)
    calibrated_probs = calibrated_probs / calibrated_probs.sum(dim=-1, keepdim=True)
    calibrated_scores = torch.log(calibrated_probs + 1e-10)
    
    print(f"\n校准后 scores[0]: {calibrated_scores[0].numpy()}")
    
    # 分析差异
    diff = (calibrated_scores - scores).abs().mean().item()
    print(f"\n平均分数差异: {diff:.6f}")
    
    pred_orig = scores.argmax(dim=1).numpy()
    pred_cal = calibrated_scores.argmax(dim=1).numpy()
    pred_changed = (pred_orig != pred_cal).sum()
    print(f"预测改变的样本数: {pred_changed} / {n_way * n_query}")
    
    print("\n" + "=" * 60)
    print("✅ 核心逻辑测试完成!")
    print("=" * 60)

if __name__ == "__main__":
    test_calibration_logic()
