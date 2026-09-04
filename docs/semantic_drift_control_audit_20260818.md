# 语义漂移闭环风格攻击：代码与论文事实审查（2026-08-18）

本文只记录可追溯事实、风险和待验证设计，不构成新颖性或性能声明。当前论文 PDF、老师建议和现有实验数字均未修改。

> 2026-08-19 复核修正：当前 `text_guide_gradient` 分支把正值 gate 乘在梯度上，随后 `fgsm_attack` 再取 `sign()`。因此对任意正常有限、非零梯度及正 gate，`sign(gate * grad) = sign(grad)`；该 gate 不改变最终 FGSM 攻击方向或由 epsilon 决定的幅度。旧文本攻击路径的实质作用主要来自 task-level epsilon scaling，而不是有效的通道级语义攻击控制。

## 1. 事实—证据—风险—修改

| 事实 | 代码/论文证据 | 风险 | 本次处理 |
|---|---|---|---|
| 现有主体继承 StyleAdv 的均值/标准差 AdaIN 风格重构、符号梯度更新和前三个 block 渐进累积 | `methods/StyleAdv_RN_GNN.py:1273-1474`；`methods/tool_func.py:23-29`；StyleAdv 官方代码与论文 | 不能把“渐进攻击”或这套风格攻击主体单独写成新贡献 | 保留旧路径；新增模块不改写继承关系 |
| 当前攻击内层使用固定 64 类 global classifier CE，而最终 FSL 任务使用 episode-local GNN query CE | `methods/StyleAdv_RN_GNN.py:115,1223-1269,1330-1342,1391-1400,1448-1457`；StyleAdv 官方论文第 4 页式 (7)-(8) 明确以 FC classifier 的 global `L_cls` 生成风格梯度；当前项目英文 PDF 却写为 `L_fsl` | 项目论文机制描述与可执行代码不一致；直接改损失会同时改变 StyleAdv 基线机制，混合两种损失则无法清楚归因 | 确定后续增加互斥的 `global_ce / episodic_ce` 开关：默认 `global_ce` 保持旧路径，最终候选方法采用 `episodic_ce`；不删除、不混合两种损失 |
| 旧 CLIP gradient gate 在正常非零梯度上被后续 `sign()` 消去，基本不改变最终 FGSM 风格扰动 | gate 由 `[0,1]` 范围的 Sigmoid 产生（`methods/style_prompt_modules.py:158-200`），在 `methods/StyleAdv_RN_GNN.py:1349-1359,1406-1415,1463-1472` 先与 style gradient 相乘；`methods/tool_func.py:23-29` 随后只使用 `data_grad.sign()`。另有 `_compute_text_guidance` 的 `@torch.no_grad()`（`methods/StyleAdv_RN_GNN.py:367`） | 仅开启 `text_guide_gradient` 时理论上近似无效；不能据此声称通道级语义攻击控制，也不能把 gate 张量“进入代码”误当成其改变了攻击。零梯度、非有限值或极端浮点饱和是实现边界，不是可依赖机制 | 将旧机制修正为“task-level 文本 epsilon 调节为主要有效分支”；保留旧 gate 仅作历史基线。新增 `weighted_sign_step` 在取符号后乘权重，首次让通道权重直接作用于单步攻击幅度，但其最终作用仍须在投影/clamp 后记录真实位移 |
| 代码使用 episode 内所有类文本的均值形成 task-level gate，而不是论文公式中的类条件 gate | `methods/StyleAdv_RN_GNN.py:393-463` | 论文中的 `g_y^(l)` 与代码行为不一致 | 审计记录，未篡改论文 |
| 代码实际为 `gradient * gate`，论文式 (11) 为 `eta * (1-g)`，架构图又写 `eta*g` | `methods/StyleAdv_RN_GNN.py:1345-1354,1402-1409,1459-1466`；英文 PDF 第 4、14 页 | 同一机制存在三种互斥表述 | 后续论文修改前必须先确定唯一实现 |
| 当前所谓视觉—文本投影是 text→visual，不是提案要求的 visual→text | `methods/StyleAdv_RN_GNN.py:299-314,619-699` | 不能复用为文本语义锚；测试时缺失投影时还会动态创建随机层 | 新增独立 visual→text anchor 原语，未接主模型 |
| episode 标签来自每个类别行的全局标签，文本名称依赖外部 category 顺序 | `data/dataset.py:73-104`；`methods/StyleAdv_RN_GNN.py:393-397`；`methods/text_prompt.py:52-95` | category 文件顺序若与 JSON 数字标签不一致，会发生静默文本错配；当前工作区无 datasets，无法实证核验 | 新 anchor 强制接收 `[0,K)` 的 episode-local 标签；主数据映射仍待真实数据审计 |
| 当前评估虽传入整集 `global_y`，但 `set_forward` 不读取它；测试时微调只用 support 生成伪样本 | `methods/StyleAdv_RN_GNN.py:734-783`；`metatest_StyleAdv_RN.py:142-176` | 在已读路径中未发现直接使用 query 真值的代码级标签泄漏；但真实 split 文件缺失，仍不能证明样本/图像无重叠 | 结论限定为“未在该调用链发现”，不宣称数据协议已通过泄漏审计 |
| StyleAdv 旧更新没有显式 `B_l` 投影；sigma 也没有正值约束 | `methods/tool_func.py:23-29` | 固定预算新攻击若实际更弱，会产生不公平的“提升”；负 sigma 可能产生无效风格统计 | 仅实现权重和单步原语；明确不冒充 `Proj_B`；为 sigma 提供最小值约束 |
| 设置 `target_dataset` 后，目标域无标签训练默认权重为 1.0 | `options.py:49,53-62`；`metatrain_StyleAdv_RN.py:457-479` | 这是 target-unlabeled/semi-supervised 协议，不能与 source-only 基线直接比较；数据不在工作区，无法验证划分泄漏 | 标为必须单独报告的协议变量 |
| 随机性设置同时启用 cuDNN deterministic 与 benchmark | `metatrain_StyleAdv_RN.py:281-285` | 不同硬件/算法下可能破坏严格复现 | 本次不做无关修改；实验前应改成 `benchmark=False` |

## 2. 近邻工作边界（原论文/官方页面）

- StyleAdv（CVPR 2023）已经包含 style mean/std 的符号梯度攻击、候选扰动率随机采样以及跨前三个 block 的 Progressive Attacking Method。因此这些是继承项。
- FAMix（CVPR 2024）已经用类别文本提示挖掘局部 class-wise style 并做 patch-wise AdaIN 混合。因此“文本引导风格”不是充分创新点。
- RASP（WACV 2024）已经朝随机错误类别优化对抗 style mean/std。因此“类别条件攻击”不是充分创新点。
- SVasP（AAAI 2025）已经聚合 crop/global style gradients，在前三个 block 渐进累积，并加入视觉差异和语义一致性目标。
- SRasP（arXiv:2603.05135, 2026 预印本）已经用全局语义选择不一致 crop、重定向并聚合 crop/global style gradients，在前三个 block 累积对抗 style，并以多目标约束视觉差异与语义一致性。它是当前必须正面对比的最近邻。
- HAP（AAAI 2026）在频域扰动幅值并保持相位，应作为不同攻击空间的近期强基线。

原始来源：

- https://openaccess.thecvf.com/content/CVPR2023/html/Fu_StyleAdv_Meta_Style_Adversarial_Training_for_Cross-Domain_Few-Shot_Learning_CVPR_2023_paper.html
- https://openaccess.thecvf.com/content/CVPR2024/html/Fahes_A_Simple_Recipe_for_Language-guided_Domain_Generalized_Segmentation_CVPR_2024_paper.html
- https://openaccess.thecvf.com/content/WACV2024/html/Kim_Randomized_Adversarial_Style_Perturbations_for_Domain_Generalization_WACV_2024_paper.html
- https://ojs.aaai.org/index.php/AAAI/article/view/33676
- https://arxiv.org/abs/2603.05135
- https://ojs.aaai.org/index.php/AAAI/article/view/39486

## 3. 论文结构与老师反馈的事实对齐

1. **标题尚未收敛**：英文 PDF 仍为 *Style-Guided Meta-Adversarial Ranking Network for Cross-Domain Few-Shot Remote-Sensing Scene Classification*，包含老师要求删除的 Meta-Adversarial、Ranking 定位，并把遥感应用场景放在了方法主定位中。
2. **引言不是要求的六段攻击因果链**：当前首段从遥感场景分类与标注成本开始，随后展开跨域遥感、RDC 和目标域关系校准；没有依次形成“攻击发展→跨分布攻击不足→文本语义攻击不足→单阶段攻击不足→三项真实贡献→结构”的六段链条。
3. **贡献仍把 RDC 作为核心组成**：第 2 页三条贡献中，第 1、3 条均包含 relational metric calibration，与老师要求删除 RDC 定位不一致。
4. **Related Work 分类轴不符合要求**：当前三个小节分别为遥感跨域小样本、风格迁移/视觉语言、目标域关系度量，不是围绕攻击形成的三类工作；SRasP、HAP 等最新近邻也尚未进入对比边界。
5. **方法标题和主图没有与拟议三模块一一对应**：当前 Overview 同时讲表示学习和 RDC，方法主体沿用 “Style-Guided Progressive Meta-Adversarial Learning”；图 1/图 2 信息密度高，且图中文字 `eta*g` 与正文式 (11) 的 `eta*(1-g)` 冲突。
6. **存在可编辑源但本次不改**：资料目录中有 `英文版(1).docx`，但代码机制、预算和训练时序尚未验证。按授权边界保留原件，未创建宣称已完成的新论文版本。

后续若改论文，只能先写“待验证方法草案”：Overview 定义渐进式攻击并解释动机，随后三个小节严格对应 Text-Grounded Semantic Anchor、Semantic-Risk-Aware Fixed-Budget Style Attack、Drift-Feedback Progressive Controller；不得复用旧表格证明新机制，也不得写入性能提升或首创声明。

## 4. 本次可安全实现的范围

新增 `methods/semantic_drift_control.py`，默认 `enabled=False`，且未接入现有训练路径：

1. **Text-Grounded Semantic Anchor 原语**：visual→text 线性投影、episode-local 文本交叉熵、冻结文本原型；攻击内层可分离 projector 参数但保留到视觉/style 变量的梯度。
2. **Semantic-Risk-Aware Fixed-Budget 原语**：按通道计算 `C*softmax(log|g_task|-beta log|g_text|)`；强制 mu/sigma 分开调用；默认 detach 权重；sigma 可做有效性下界。
3. **Drift-Feedback 原语**：显式传递 detached lambda 状态，计算阶段 Lagrangian，并按实测 drift 做投影更新。

这只是可执行机制骨架，不是完整方法，也没有任何性能证据。

## 5. 完整接入前必须做出的设计选择

1. **内层任务损失（已确定）**：最终候选方法使用 episode-local GNN/FSL query CE，同时保留 global 64-class CE 作为 StyleAdv 对照分支。两者必须通过 `global_ce / episodic_ce` 互斥开关分别运行，不删除 global CE，也不混合损失。`global_ce` 只能称为“严格 StyleAdv 复现”条件的一部分；还必须关闭文本 epsilon/gate、新三模块及其他额外训练路径，并保持预算、外层目标、数据协议和随机种子一致。
2. **视觉→文本投影训练时序**：建议候选方案至少包括“先用 clean outer loss 预热后冻结”和“outer 更新、每次 inner attack 使用参数快照”。随机初始化投影不能直接当语义边界。
3. **attack set `B_l`**：选择与 StyleAdv 候选 epsilon 匹配的 L-infinity 约束，或固定 L1 总位移；必须同时记录理论预算和 clamp 后真实位移。二者不能混称相同预算。
4. **drift 聚合**：每样本、episode mean、max 或分位数会产生不同约束强度，不能默认替用户选择。
5. **lambda 策略**：初始化、上界、`rho`、跨 episode 是否重置尚未确定。当前原语默认每 episode 显式初始化并跨 block stop-gradient。
6. **计算成本**：每层同时求 task/risk gradients，并把当前攻击特征走完剩余 blocks，至少增加额外 autograd 和前向；需先做单 episode 显存/时间剖析，不能直接跑完整训练。

## 6. 后续最小实验矩阵

所有条件必须使用相同数据划分、总攻击预算、计算量口径、随机种子和评估协议：

1. StyleAdv + `global_ce`；StyleAdv 攻击机制 + `episodic_ce`；`episodic_ce` + 语义锚/约束；+固定总预算风险分配；+漂移反馈完整机制。第 1→2 项隔离内层损失变化，第 2→5 项衡量新语义机制的总体增量；逐模块归因分别看第 2→3、3→4、4→5 项。
2. correct text / shuffled labels / random text。
3. uniform / random / equal-budget / semantic-risk weights。
4. fixed lambda / feedback lambda。
5. 每阶段记录 `d_1,d_2,d_3`、task loss、clean/attacked accuracy、理论与真实攻击位移、运行时间和峰值显存。
6. 与 StyleAdv、SVasP、SRasP、HAP 按完全相同协议比较；target-unlabeled 与 source-only 结果必须分表。

证伪标准保持严格：shuffled≈correct 表示文本语义分支可能无效；uniform≈risk 表示风险预算可能只是装饰；fixed lambda≈feedback 且漂移曲线不变，则不能声称闭环渐进控制。
