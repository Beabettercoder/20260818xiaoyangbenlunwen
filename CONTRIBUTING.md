# 代码与实验记录规范

本仓库采用“小步修改、单一目的、先验证后提交”的工作方式。每个提交都应能回答：为什么改、改了什么、如何验证、是否改变实验协议。

## 每次修改的固定流程

1. 更新主分支并确认工作区状态：

   ```bash
   git switch main
   git pull --ff-only
   git status
   ```

2. 为一个明确目的创建分支，例如：

   ```bash
   git switch -c feat/semantic-anchor
   ```

   推荐前缀：`feat/`、`fix/`、`test/`、`exp/`、`docs/`。

3. 修改后先检查差异和目标测试：

   ```bash
   git diff
   git diff --check
   python -m py_compile <本次涉及的 Python 文件>
   python -m unittest <本次涉及的测试>
   ```

4. 只暂存本次修改的明确文件，不习惯性使用 `git add .`：

   ```bash
   git add path/to/file1 path/to/file2
   git diff --cached
   ```

5. 提交信息使用“类型: 目的”：

   ```bash
   git commit -m "feat: add frozen inner-loop semantic anchor"
   ```

   推荐类型：`feat`、`fix`、`test`、`docs`、`exp`、`refactor`、`chore`。

6. 推送分支，并把提交号、命令、结果和结论写入 `docs/EXPERIMENT_LOG.md`：

   ```bash
   git push -u origin feat/semantic-anchor
   ```

## 实验提交必须写清楚

- 科研目的与假设；
- 唯一变化项；
- 保持不变的基线配置；
- 运行过的语法检查和测试；
- 数据协议、随机种子、训练轮数是否改变；
- 已知风险和未验证结论。

## 禁止进入 GitHub 的内容

- 数据集和数据副本；
- 模型权重、特征缓存和中间张量；
- 原始训练日志、输出目录和临时文件；
- 密钥、令牌、密码、个人绝对路径；
- 未经脱敏的会话记录和个人文档。

论文中已经核验的少量汇总结果可以写入 Markdown；原始日志和权重保留在服务器，并在实验登记中记录服务器相对位置。

## 投稿版本

只有在代码冻结、主表与消融复核、测试通过后才创建投稿标签，例如：

```bash
git tag -a submission-v1 -m "Frozen code for submission experiments"
git push origin submission-v1
```

禁止覆盖已有提交或强制推送主分支。
