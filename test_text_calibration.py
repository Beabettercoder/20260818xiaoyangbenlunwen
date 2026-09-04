#!/usr/bin/env python3
"""
测试脚本：验证 Test-time Text Calibration 功能
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_text_calibration(data_root):
    print("=" * 60)
    print("测试: Test-time Text Calibration")
    print("=" * 60)
    
    from methods.StyleAdv_RN_GNN import StyleAdvGNN
    from methods.backbone_multiblock import model_dict
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 创建模型 - 启用 text_calibration
    print("\n--- 创建模型 (text_calibration_weight=0.2) ---")
    model = StyleAdvGNN(
        model_dict['ResNet10'],
        n_way=5,
        n_support=5,
        dataset_name='NWPU',
        data_root=data_root,
        text_guide_epsilon=1,
        text_guide_gradient=1,
        use_prompt_ensemble=True,
        text_calibration_weight=0.2,
    ).to(device)
    model.eval()
    
    print(f"text_calibration_weight: {model.text_calibration_weight}")
    print(f"text_prompt_encoder: {'OK' if model.text_prompt_encoder is not None else 'None'}")
    
    # 模拟 feature-based 评估
    print("\n--- 模拟 feature-based 评估 ---")
    n_way, n_support, n_query = 5, 5, 15
    feat_dim = 512  # ResNet10 feature dim
    
    # 模拟提取的特征 (is_feature=True 模式)
    # z_all: [n_way, n_support+n_query, feat_dim]
    z_all = torch.randn(n_way, n_support + n_query, feat_dim).to(device)
    
    # 模拟类名 (从 HDF5 的 key)
    class_names = ['airport', 'bridge', 'harbor', 'meadow', 'river']
    print(f"Class names: {class_names}")
    
    # 测试 set_forward
    model.n_query = n_query
    
    print("\n--- 测试 without text calibration ---")
    model.text_calibration_weight = 0.0
    with torch.no_grad():
        scores_no_cal = model.set_forward(z_all, is_feature=True, class_names=class_names)
    print(f"Scores shape: {scores_no_cal.shape}")
    print(f"Scores sample: {scores_no_cal[0].cpu().numpy()}")
    
    print("\n--- 测试 with text calibration (w=0.2) ---")
    model.text_calibration_weight = 0.2
    with torch.no_grad():
        scores_with_cal = model.set_forward(z_all, is_feature=True, class_names=class_names)
    print(f"Scores shape: {scores_with_cal.shape}")
    print(f"Scores sample: {scores_with_cal[0].cpu().numpy()}")
    
    # 比较差异
    print("\n--- 分析差异 ---")
    diff = (scores_with_cal - scores_no_cal).abs().mean().item()
    print(f"平均分数差异: {diff:.6f}")
    
    # 检查预测是否改变
    pred_no_cal = scores_no_cal.argmax(dim=1).cpu().numpy()
    pred_with_cal = scores_with_cal.argmax(dim=1).cpu().numpy()
    pred_changed = (pred_no_cal != pred_with_cal).sum()
    print(f"预测改变的样本数: {pred_changed} / {n_way * n_query}")
    
    # 测试文本嵌入获取
    print("\n--- 验证文本嵌入 ---")
    text_emb = model._get_text_embeddings_for_names(class_names)
    if text_emb is not None:
        print(f"Text embeddings shape: {text_emb.shape}")
        # 计算类间相似度
        text_emb_norm = torch.nn.functional.normalize(text_emb, dim=-1)
        sim_matrix = torch.mm(text_emb_norm, text_emb_norm.t())
        print(f"Text similarity matrix:\n{sim_matrix.cpu().numpy()}")
    
    print("\n" + "=" * 60)
    print("✅ Text Calibration 测试完成!")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke-test the test-time text calibration path.")
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Dataset root containing one directory per dataset.",
    )
    args = parser.parse_args()
    test_text_calibration(args.data_dir)
