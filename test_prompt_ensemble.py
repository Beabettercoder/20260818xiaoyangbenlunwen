#!/usr/bin/env python3
"""
测试脚本：验证多模板集成是否产生不同的文本嵌入
"""
import torch
import sys
sys.path.insert(0, '.')

from methods.style_prompt_modules import TextPromptEncoder
from methods.text_prompt import get_ensemble_templates, default_template

def test_prompt_ensemble():
    print("=" * 60)
    print("测试: 单模板 vs 多模板集成的文本嵌入差异")
    print("=" * 60)
    
    # 初始化编码器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = TextPromptEncoder(clip_model_name="ViT-B/32", device=device)
    
    # 测试类别
    dataset_name = "NWPU"
    class_names = ["airport", "bridge", "harbor"]
    
    print(f"\n数据集: {dataset_name}")
    print(f"测试类别: {class_names}")
    
    # 方法1: 单模板
    print("\n" + "-" * 60)
    print("方法1: 单模板")
    print("-" * 60)
    template = default_template(dataset_name)
    print(f"模板: {template}")
    single_prompts = [template.format(name=name) for name in class_names]
    print(f"生成的prompts: {single_prompts}")
    
    class_emb_single, z_text_single = encoder.encode_class_prompts(single_prompts)
    print(f"class_embeddings shape: {class_emb_single.shape}")
    print(f"z_text shape: {z_text_single.shape}")
    print(f"z_text norm: {z_text_single.norm().item():.6f}")
    print(f"z_text[:10]: {z_text_single[:10].tolist()}")
    
    # 方法2: 多模板集成
    print("\n" + "-" * 60)
    print("方法2: 多模板集成 (8 templates)")
    print("-" * 60)
    templates = get_ensemble_templates(dataset_name)
    print(f"模板数量: {len(templates)}")
    print(f"模板示例:")
    for i, t in enumerate(templates[:3], 1):
        print(f"  {i}. {t}")
    print(f"  ...")
    
    class_emb_ensemble, z_text_ensemble = encoder.encode_class_prompts_ensemble(
        class_names, templates
    )
    print(f"class_embeddings shape: {class_emb_ensemble.shape}")
    print(f"z_text shape: {z_text_ensemble.shape}")
    print(f"z_text norm: {z_text_ensemble.norm().item():.6f}")
    print(f"z_text[:10]: {z_text_ensemble[:10].tolist()}")
    
    # 计算差异
    print("\n" + "=" * 60)
    print("差异分析")
    print("=" * 60)
    
    # L2距离
    l2_dist = (z_text_single - z_text_ensemble).norm().item()
    print(f"z_text L2 距离: {l2_dist:.6f}")
    
    # 余弦相似度
    cos_sim = torch.nn.functional.cosine_similarity(
        z_text_single.unsqueeze(0), 
        z_text_ensemble.unsqueeze(0)
    ).item()
    print(f"z_text 余弦相似度: {cos_sim:.6f}")
    
    # 类级别差异
    class_l2_dists = (class_emb_single - class_emb_ensemble).norm(dim=1)
    print(f"\n类级别 L2 距离:")
    for i, (name, dist) in enumerate(zip(class_names, class_l2_dists)):
        print(f"  {name}: {dist.item():.6f}")
    
    # 判断
    print("\n" + "=" * 60)
    if l2_dist < 1e-6:
        print("❌ 警告: 嵌入几乎完全相同！多模板集成可能没有生效！")
    elif l2_dist < 0.01:
        print("⚠️  差异很小 (L2 < 0.01)，可能影响有限")
    else:
        print(f"✅ 多模板集成产生了显著差异 (L2 = {l2_dist:.6f})")
    print("=" * 60)

if __name__ == "__main__":
    test_prompt_ensemble()
