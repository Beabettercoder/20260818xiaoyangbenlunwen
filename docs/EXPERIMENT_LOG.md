# 实验登记

每次正式或诊断实验都新增一条记录。不要覆盖旧记录，不要只保留“最好的一次”。

## 登记模板

```text
实验 ID：
状态：planned / running / completed / invalid
日期：
科研目的：
Git commit：
唯一变化：
源域 / 目标域：
协议：source-only / target-unlabeled
way / shot / query：
训练随机种子：
测试随机种子：
warmup checkpoint：
训练命令或脚本：
环境：Python / PyTorch / CUDA / GPU
服务器日志位置：
结果：
结论：
风险或偏离：
```

## BASE-SO-NWPU-5S-001

- 状态：completed（历史结果，尚未绑定当前 Git commit）。
- 科研目的：恢复 NWPU source-only 5-shot 基线。
- 源域 / 目标域：NWPU / EuroSAT、UCM、AID。
- 协议：source-only，5-way 5-shot，15 query/class，1000 test episodes。
- warmup：`output/checkpoints/baseline/399.tar`。
- 服务器结果：`output/checkpoints/nwpu_5shot_baseline_restore/acc_bscdfsl.txt`。
- EuroSAT：90.73% ± 0.39%。
- UCM：98.15% ± 0.20%。
- AID：97.07% ± 0.25%。
- 结论：这是目前可信的最佳恢复基线；投稿前需在冻结代码上复跑并补齐训练种子。

## TSSL-NWPU-AID-DIAG-001

- 状态：invalid。
- 科研目的：确认目标域无标签训练链路是否真正产生有效监督。
- 观察：默认设置下 `pseudo_confidence≈0.0241`、`mean_entropy≈0.9955`、`keep_ratio=0`、`target_ssl=0`。
- 放宽全部筛选后：`keep_ratio=1`，但置信度和熵基本不变。
- 结论：链路可以执行，但伪标签接近随机；不得作为有效目标域无标签结果。

## SEMANTIC-CONTROL-UNIT-001

- 状态：completed。
- Git commit：`bf9c7ac`。
- 科研目的：验证语义锚、独立预算和漂移反馈的计算图原语。
- 检查：语法检查通过，`tests.test_semantic_drift_control` 共 11 项测试通过。
- 覆盖：正确/打乱文本分支、mu/sigma 独立权重、stop-gradient、lambda 更新、sigma 下界、风格变量梯度和关闭开关。
- 结论：实现具备进入 smoke test 的条件；单元测试不能证明分类精度有效。

## SUBMISSION-QUEUE-20260904

- 状态：planned。
- 当前代码事实：三个渐进攻击阶段仍由固定全局分类器 CE 产生任务梯度，尚无 `global_ce / episodic_ce` 互斥开关。
- 当前候选命名：`M-G = global CE + semantic anchor + budget + feedback`，不得误记为 M4。
- 执行设备：单张物理 GPU 4，所有实验串行。
- 执行入口：`scripts/run_submission_source_only.sh`。
- 执行顺序：M-G 5-shot → M-G 1-shot → 同提交补齐 B0 1-shot。
- 协议：只用 NWPU 训练；每个 shot 训练一次，并用同一个 checkpoint 测试 AID、UCM、EuroSAT。
- 80/20 目标无标签旧运行继续标记为 invalid：默认 `keep_ratio=0`；全放开后 `pseudo_conf≈0.0241`、`mean_entropy≈0.9955`，伪标签近似随机。
