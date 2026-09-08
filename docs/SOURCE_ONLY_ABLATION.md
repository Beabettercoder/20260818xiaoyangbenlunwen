# Source-only A–E ablation, revision 2

This is a new controlled experiment, not a relabelling of historical checkpoints.
The historical `nwpu_5shot_baseline_restore` score is not evidence that the
repaired text-scale mechanism was enabled. Do not copy that score into arm E.

| Arm | Style attack | Text-conditioned candidate scales |
| --- | --- | --- |
| A | Disabled | Off |
| B | Independent | Off |
| C | Progressive | Off |
| D | Independent | On |
| E | Progressive | On |

Independent means inner attack gradients use clean stage inputs. Both modes
propagate all three stage perturbations through the outer adversarial branch.
Compare B−A, C−B, D−B, E−C and E−D; C→D changes two components.

## Fixed settings

- NWPU source only; no target images used for training. Target SSL, semantic
  anchor/drift control, gradient gates and test prototype refinement are off.
- Every arm starts from the same `baseline/399.tar`, NOT the best fine-tuned model.
- ResNet-10, 5-way, separate 5-shot and 1-shot training; training query count 5,
  evaluation query count 15; seed 0 for training and target episode sampling.
- 200 epochs, 100 train and validation episodes per epoch, 1000 test episodes
  per target (AID, UCM, EuroSAT). Existing source validation selects the checkpoint.
- Loss coefficients are inherited unchanged. A has clean global and episode CE;
  its disabled adversarial branch contributes no adversarial CE and zero KL.
  B–E retain clean/adversarial episode CE, episode KL and clean global CE.
- Candidate scales inherit the implementation; text scale range is [0.3, 1.7].
  The inherited 16/255 initialization shift and 0.4 outer style-skip probability
  remain unchanged. The initialization shift is constant, not random noise.

## Changes and checks

Removed the no-grad scope over the conditioning heads and scalar `.item()`
conversion of their outputs. Text embeddings remain detached/frozen. Inner
first-order attacks use `autograd.grad` for style statistics only; previous inner
stage statistics are detached without disconnecting outer scale gradients.
Independent mode no longer discards earlier outer stage perturbations.

Added optional `--eval_seed` (default None for backward compatibility). The runner
sets it to 0 and each target resets RNGs after model construction, before episode
sampling. The inherited class-balanced, label-organized GNN evaluation is not
redesigned here; this change is not a resolution of the query-grouping protocol
question. Use the same evaluation implementation for all arms.

CPU regression tests exercise the actual extracted attack/guidance methods with
a small backbone in both modes: finite nonzero scale-head gradients, an optimizer
update, and no text-embedding gradients. These tests are NOT a full ResNet/GNN
CUDA training validation. Run the server smoke test before formal training.

## Server setup (one-time, outside the old training directory)

The separate clone preserves existing server code, checkpoints and results.

```bash
git clone --branch fix/source-only-ablation --single-branch https://github.com/Beabettercoder/20260818xiaoyangbenlunwen.git /mnt/sdc/wzj/SGA-Net-ablation-v2
mkdir -p /mnt/sdc/wzj/SGA-Net-ablation-v2/output/checkpoints/baseline
cp -n /mnt/sdc/wzj/SGA-Net/output/checkpoints/baseline/399.tar /mnt/sdc/wzj/SGA-Net-ablation-v2/output/checkpoints/baseline/399.tar
tmux new -s wzj_ablation_v2
```

Inside the new tmux, paste the following block. It uses physical GPU **4 only**,
one process at a time. Formal training starts only if both text-arm smoke runs
and their target evaluations succeed. Smoke scores are not experimental results.

```bash
cd /mnt/sdc/wzj/SGA-Net-ablation-v2 && \
/mnt/sdc/wzj/envs/sganet/bin/python3 scripts/run_source_only_ablation.py \
  --data-dir /mnt/sdc/wzj/datasets --gpu 4 --arms D E --shots 5 --smoke && \
/mnt/sdc/wzj/envs/sganet/bin/python3 scripts/run_source_only_ablation.py \
  --data-dir /mnt/sdc/wzj/datasets --gpu 4
```

The default formal run trains ten models (five arms × two shots) and produces
thirty target results. Do not launch a duplicate copy. Detach with Ctrl+B then D;
reconnect using `tmux attach -t wzj_ablation_v2`. The server must stay powered on.

## Reproducibility and results

`logs/source_only_ablation/formal_<timestamp>/results.csv` contains per-arm,
per-shot, per-target results and reported uncertainty (without assuming its
statistical definition). `manifest.json` records commands, code commit/status,
code/split SHA-256 hashes, full warmup and final checkpoint hashes and run states.
Logs and checkpoints are never uploaded by this workflow.

Missing warmup, unexpected later warmup epoch, existing run directories, failed
subprocesses or incomplete evaluations stop execution. Optional
`--warmup-sha256 FULL_64_CHARACTER_HASH` verifies an independently recorded hash.
A failed run may retain a training/testing state; consult its log before retrying
with a new tag. Automatic checkpoint resume is deliberately not implemented.

For a no-training preview, append `--dry-run`. Local checks:

```bash
python tests/test_ablation_gradient_path.py
python -m py_compile methods/StyleAdv_RN_GNN.py options.py test_function_bscdfsl_benchmark.py scripts/run_source_only_ablation.py
git diff --check
```

Only after the actual thirty results are available, plot two panels (1-shot and
5-shot), three target curves and categorical A–E ticks. Preserve decreases;
do not smooth curves or substitute historical best scores. This single-seed
experiment does not establish training-seed robustness.
