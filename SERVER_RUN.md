# 服务器运行说明

本文档只使用仓库相对路径和命令行参数，不假定仓库位于某个固定目录。以下命令在服务器终端执行，/server/... 只是示例路径，请替换为实际路径。

## 1. 数据目录

数据根目录通过 --data_dir 传入，目录结构至少应为：

    DATA_ROOT/
      NWPU/
        base.json
        val.json
        novel.json
        unlabeled.json       # 严格跨域训练协议需要时提供
        <images or image folders>/
      AID/
      UCM/
      EuroSAT/

JSON 中如果仍保存着旧电脑的绝对图片路径，加载器会在提供 --data_dir 后按“数据集名称 + 相对后缀”尝试重定位；如果无法重定位，会明确报告找不到的图片。建议迁移服务器前先用数据检查脚本确认。

## 2. 环境安装

    cd /path/to/xiaoyangbenlunwen
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    # 先按服务器 CUDA 版本安装匹配的 torch/torchvision
    python -m pip install torch torchvision
    python -m pip install -r requirements-server.txt

CLIP 第一次运行可能下载 ViT-B/32 权重。若服务器不能联网，应提前缓存权重，或关闭需要 CLIP 的文本模块。

## 3. 检查数据和划分

    python scripts/check_dataset_splits.py /server/datasets --source NWPU --n-way 5
    python scripts/audit_split_leakage.py --data_dir /server/datasets
    python scripts/test_dataloader.py --dataset NWPU --data_dir /server/datasets

如果要按当前严格 pairwise 协议重新生成划分，先备份数据目录，再执行：

    python scripts/restructure_datasets.py \
      --data-dir /server/datasets \
      --source NWPU \
      --protocol strict_pairwise \
      --source-train-ratio 0.8 \
      --target-unlabeled-ratio 0.2

该协议使用全部源域标注数据，并使用目标域 20% 无标签子集参与训练；目标域剩余 80% 只用于评估。目标 query 真值不能参与训练或控制器更新。

## 4. 主训练、测试和微调

通用脚本的第一个参数永远是数据根目录，之后的参数原样传给 Python 主入口：

    bash scripts/run_train.sh /server/datasets \
      --source_dataset NWPU \
      --target_dataset EuroSAT \
      --name NWPU_to_EuroSAT_5shot \
      --n_shot 5 \
      --semantic_anchor 1 \
      --semantic_drift_control 1

    bash scripts/run_test.sh /server/datasets \
      --source_dataset NWPU \
      --dataset EuroSAT \
      --name NWPU_to_EuroSAT_5shot \
      --n_shot 5

    bash scripts/run_finetune.sh /server/datasets \
      --source_dataset NWPU \
      --testset EuroSAT \
      --resume_dir NWPU_to_EuroSAT_5shot \
      --n_shot 5

默认输出写入仓库下的 output/；可用 --save_dir /server/output 改到其他位置。测试和微调使用的检查点应位于对应 save_dir/checkpoints/<name>/ 目录中。

多卡训练可直接调用主入口：

    torchrun --nproc_per_node=4 metatrain_StyleAdv_RN.py \
      --data_dir /server/datasets \
      --source_dataset NWPU \
      --target_dataset EuroSAT \
      --name NWPU_to_EuroSAT_5shot

## 5. 服务器迁移边界

slurm/ 下按日期保存的文件是历史实验记录，其中可能保留旧服务器路径、旧检查点和旧实验参数；它们不作为通用启动入口。服务器上请优先使用 scripts/run_train.sh、scripts/run_test.sh 和 scripts/run_finetune.sh，或者复制历史脚本后自行替换路径和参数。

论文画图脚本优先使用仓库内 fonts/times.ttf、fonts/timesbd.ttf，也可以用 TSNE_PAPER_FONT、TSNE_PAPER_FONT_BOLD 或 CONFUSION_PAPER_FONT 指定字体文件。
