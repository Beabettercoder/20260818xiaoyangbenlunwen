# Semantic-Guided Progressive Adversarial Learning

本仓库保存跨域小样本分类论文的可复现代码、目标测试、实验协议和修改记录。当前研究主线是：利用文本语义锚、通道风险预算和跨阶段语义漂移反馈，改进渐进式风格对抗训练。

## 当前真实状态

- 源域：NWPU。
- 目标域：AID、UCM、EuroSAT。
- 主表协议：source-only cross-domain few-shot learning。
- 评估设置：5-way，1-shot/5-shot，每类 15 个 query，1000 个测试 episode。
- 语义锚、独立风险预算和漂移反馈已接入训练代码，并通过语法检查与 11 项单元测试。
- 新机制的正式精度增益尚未验证，不能提前写成有效结论。
- 目标域无标签路径可以运行，但当前伪标签接近随机，暂不作为论文主结果。

## 当前可信的恢复基线

以下为 NWPU source-only、5-way 5-shot、1000 个测试 episode 的历史核验结果：

| Target | Accuracy |
| --- | ---: |
| EuroSAT | 90.73% ± 0.39% |
| UCM | 98.15% ± 0.20% |
| AID | 97.07% ± 0.25% |

这里的 `±` 为测试 episode 上的 95% 置信区间，不等同于多个训练随机种子的标准差。1-shot 对齐基线、最终方法和模块消融仍需在同一冻结代码版本上正式运行。

## 投稿前实验

- [最小必要实验计划](docs/EXPERIMENT_PLAN.md)
- [实验登记](docs/EXPERIMENT_LOG.md)
- [代码与实验记录规范](CONTRIBUTING.md)
- [服务器运行说明](SERVER_RUN.md)

## 核心入口

- 训练：`metatrain_StyleAdv_RN.py`
- 跨域评估：`test_function_bscdfsl_benchmark.py`
- 模型主体：`methods/StyleAdv_RN_GNN.py`
- 语义漂移控制：`methods/semantic_drift_control.py`
- 参数定义：`options.py`
- 目标单测：`tests/test_semantic_drift_control.py`

## 最小验证

```bash
python -m py_compile \
  options.py \
  metatrain_StyleAdv_RN.py \
  methods/StyleAdv_RN_GNN.py \
  methods/meta_template_StyleAdv_RN_GNN.py \
  methods/semantic_drift_control.py \
  test_function_bscdfsl_benchmark.py

python -m unittest tests.test_semantic_drift_control -v
```

## 版本原则

- 一个科研目的对应一个分支和一个可解释提交。
- 数据、权重、日志、特征缓存和个人文档不进入 GitHub。
- 每个实验登记 Git commit、协议、随机种子、命令、日志位置、结果和风险。
- 只有完成主表复核和必要消融后，才创建投稿版本标签。
