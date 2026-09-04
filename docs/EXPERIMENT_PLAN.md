# 投稿前最小必要实验计划

## 要回答的三个核心问题

1. 在完全相同的训练和评估协议下，最终方法是否优于恢复后的 source-only 基线？
2. 增益是否来自正确的文本语义，而不是额外参数、随机权重或攻击强度变化？
3. 语义锚、风险预算和漂移反馈是否各自带来可解释的增量？

## 固定协议

- 源域：NWPU。
- 目标域：AID、UCM、EuroSAT。
- 任务：5-way 1-shot 和 5-way 5-shot。
- 每类 query：15。
- 正式评估：每个目标域 1000 个 episode，报告均值与 95% 置信区间。
- 主表协议：source-only。目标域测试数据不得用于训练、调参或选择 checkpoint。
- 所有比较固定 warmup、backbone、数据划分、优化器、训练轮数、训练 episode 数和测试采样规则。
- 使用目标域 20% 无标签数据的实验必须单独成表，不能与 source-only 混合比较。

## P0：投稿前必须完成

### B0：冻结代码后的 source-only StyleAdv-global 基线

- 1-shot：AID、UCM、EuroSAT。
- 5-shot：AID、UCM、EuroSAT。
- 当前 5-shot 历史结果只能作为参考，最终表应在冻结代码上复跑。
- 关闭新增文本机制，但保留原 StyleAdv 渐进风格攻击；当前攻击内层损失是固定全局分类头 CE，实验名记为 `styleadv_global`。

### M-G：当前代码可运行的完整语义控制版本

配置：global CE + 训练后的语义锚 + 固定预算风险分配 + 漂移反馈。

- 先做 1 个 epoch 的 smoke test，验证梯度、checkpoint 和评估链路。
- 然后按 B0 的全部 1-shot/5-shot 设置运行。
- 时间允许时，B0 与 M-G 各使用 3 个训练随机种子；1000 个测试 episode 不能替代多个训练种子。

当前仓库尚未实现 `global_ce / episodic_ce` 攻击内层互斥开关，因此 M-G 只能先回答“新文本机制在不改变 StyleAdv 攻击损失时是否有效”。在 episodic 分支实现并通过梯度测试前，不得把 M-G 写成 episodic 攻击版本。

### 文本因果对照

除文本输入外配置完全一致：

- T0：关闭文本语义，使用 uniform/equal-budget control。
- T1：使用正确类别文本。
- T2：随机打乱类别文本映射。

如果 T1 不能稳定优于 T0 和 T2，论文不能声称提升来自文本语义。

## 单卡 GPU 4 的立即执行顺序

只有物理 GPU 4 可用，所有任务必须串行：

1. M-G 5-shot smoke test。
2. M-G 5-shot 正式训练与三个目标域测试。
3. M-G 1-shot 正式训练与三个目标域测试。
4. 在相同 Git 提交上补齐 B0 1-shot；历史 B0 5-shot暂作参照，最终表再复跑。
5. 若 M-G 没有稳定优于 B0，先检查语义漂移、预算利用率和 `sigma_clamp_rate`，不扩大实验矩阵。
6. 主结果成立后，再实现 episodic CE、shuffled text 和逐模块消融。

统一入口：

```bash
export PATH=/mnt/sdc/wzj/envs/sganet/bin:$PATH
cd /mnt/sdc/wzj/SGA-Net

# 只核对命令，不启动训练
DRY_RUN=1 bash scripts/run_submission_source_only.sh \
  /mnt/sdc/wzj/datasets semantic_global "5 1" 4

# 真实 smoke test：1 epoch、1 个训练 episode、2 个测试 episode
EPOCHS=1 TRAIN_EPISODES=1 VAL_EPISODES=2 TEST_EPISODES=2 \
RUN_TAG=smoke_gpu4 bash scripts/run_submission_source_only.sh \
  /mnt/sdc/wzj/datasets semantic_global 5 4

# 先 5-shot，自动测完三个目标域，再运行 1-shot
bash scripts/run_submission_source_only.sh \
  /mnt/sdc/wzj/datasets semantic_global "5 1" 4
```

脚本不传 `target_dataset`，目标 SSL 权重固定为 0。每个 shot 只训练一次 NWPU，然后使用同一个 checkpoint 依次测试 AID、UCM、EuroSAT。

## P1：模块消融

严格按以下顺序增加模块：

| 编号 | 配置 | 归因 |
| --- | --- | --- |
| A0 | StyleAdv + global CE，关闭所有新增文本机制 | 旧攻击基线 |
| A1 | StyleAdv + episodic CE，关闭语义锚 | 攻击损失变化 |
| A2 | A1 + 训练后的语义锚 | 语义锚增量 |
| A3 | A2 + 固定预算风险分配 | 风险预算增量 |
| A4 | A3 + 漂移反馈 | 反馈增量/完整方法 |

时间紧张时，消融可先用 1 个训练种子，但至少覆盖三个目标域的 1-shot；B0 与 A4 优先完成多训练种子。A1-A4 当前仍是待实现队列，不得用 M-G 冒充。

## 机制诊断（随正式实验记录）

- 各阶段 clean/attacked 文本损失及语义漂移；
- 各阶段 lambda；
- mu、sigma 的 `mean_abs_delta`、`linf_delta` 和预算利用率；
- `sigma_clamp_rate`；
- support/query 漂移均值、样本级漂移分布和正漂移比例。

所有预算和漂移指标必须基于投影、裁剪后的实际变量，而不是生成攻击梯度时的候选值。

## 当前不能写入论文主结果的实验

目标域无标签分支虽然能够运行，但现有诊断中默认阈值下出现：

- `pseudo_confidence` 约为 0.024；
- `mean_entropy` 约为 0.996；
- `keep_ratio` 接近 0；
- `target_ssl` 实际为 0。

把阈值完全放开后虽可令 `keep_ratio=1`，但低置信度没有改善，只是把接近随机的伪标签全部纳入训练。因此该路径目前只能标记为无效诊断，不能声称已经完成“80% source + 20% unlabeled target”有效训练。

## 停止条件

- 完整方法不优于冻结基线：先检查机制与预算，不继续堆模块。
- 正确文本不优于 shuffled text：停止文本因果主张。
- 实验协议、代码提交号或随机种子缺失：结果不得进入论文表格。
