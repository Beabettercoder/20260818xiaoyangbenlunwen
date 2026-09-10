# Nine source-only transfers

Entry point: `scripts/run_ablation_nine.py`. GPU defaults to **6**, sequential.

Before filling the source configuration, run `python scripts/inventory_ablation_inputs.py`
on the server. It records available split sizes and candidate checkpoints under
`/mnt/sdc/wzj` in a timestamped `logs/ablation_inputs_*.json`. It does not train,
change splits, or assume checkpoint provenance. Missing source-specific inputs
remain blockers, not automatically substituted NWPU weights.
Sources NWPU/AID/UCM each test the other three datasets (including EuroSAT).
Only A/B/C/D are scheduled; E is already available and is not rerun or merged.
Both shots: 18 training jobs, 72 evaluations. No target data enters training.

## Locked A-D setup (2026-09-10)

A evaluates the existing source warmup directly: zero additional epochs.
B/C/D each start from the SAME warmup checkpoint (not from the previous arm).
The schedule explicitly pins ResNet10, baseline method, 5-way, 200 epochs,
100 train/validation episodes, 5 training queries/class, 15 evaluation
queries/class, and attack candidates 0.8/0.08/0.008. Test episodes are 1000.
Only independent/progressive attack and text-scale switches differ across B-D.
Target SSL, semantic anchor/drift and text-gradient gating remain disabled.
GPU6 is used sequentially. The loader-only speed settings are fixed at
`train_workers=4`, `eval_workers=4`, `feature_batch_size=64`, and
`prefetch_factor=2`. These change input pipeline throughput only; the actual
training episode remains 5-way with the same support/query counts. They are
recorded in the run manifest and can be overridden if server CPU/RAM requires
it. E is rejected, not restored.

The audited `nwpu_5shot_baseline_restore` checkpoint only records dataset,
model, method, ways, shot and name. It has no text parameter keys and does not
record the complete training configuration. Consequently exact equivalence
to every hyperparameter of that historic run is NOT verified. The manifest
records this limitation and the audited reference hash; it is not a runtime
verification of a locally available reference checkpoint. Current Adam defaults
and existing loss implementation remain unchanged. A is a warmup reference,
not a budget/head-matched module ablation against B-D.

A directly evaluates the source-pretrained ResNet10 with cosine prototypes;
it is NOT random initialization and receives no further training. B independent
attack; C progressive attack; D independent + text scales; E progressive + text
scales. B–E load the corresponding source checkpoint and use the existing GNN.
Therefore A–B changes training budget AND prediction head, not just attack.
Use B/C/D/E for component comparisons. No best-score ordering is guaranteed.
Existing GNN balanced-query grouping is unchanged; this script does not resolve
that evaluation-protocol limitation. Do not describe it as label-independent grouping.

The requested protocol intentionally reuses the single verified
`baseline/399.tar` for all three source names.  First build complete source
roots from the real image lists.  The builder keeps NWPU's existing base/val,
splits AID/UCM class-wise with the same validation proportion (while enforcing
the benchmark minimum), makes source `novel.json` empty, and puts all real
non-source images in target `novel.json`:

```bash
python scripts/build_ablation_source_splits.py \
  --input-root /mnt/sdc/wzj/datasets \
  --output-root /mnt/sdc/wzj/datasets_causal_splits/ablation_source_splits \
  --reference-source NWPU --sources NWPU AID UCM --seed 0 \
  --verify-images
```

Use a new output-root if that directory already exists.  The generated
`split_manifest.json` records the exact counts and policy.  Create a JSON
config automatically so that no placeholder path can be copied into the run:

```bash
python scripts/make_ablation_shared_config.py \
  --split-root /mnt/sdc/wzj/datasets_causal_splits/ablation_source_splits \
  --checkpoint /mnt/sdc/wzj/SGA-Net-ablation-v2/output/checkpoints/baseline/399.tar \
  --output /mnt/sdc/wzj/SGA-Net-ablation-v2/logs/ablation_sources_shared.json
```

The generated JSON has three entries, each using **verified real paths**:

```json
{
  "NWPU": {"data_dir": "/mnt/sdc/wzj/datasets_causal_splits/ablation_source_splits/NWPU_source", "checkpoint": "/mnt/sdc/wzj/SGA-Net-ablation-v2/output/checkpoints/baseline/399.tar", "pretrained_source": "shared"},
  "AID": {"data_dir": "/mnt/sdc/wzj/datasets_causal_splits/ablation_source_splits/AID_source", "checkpoint": "/mnt/sdc/wzj/SGA-Net-ablation-v2/output/checkpoints/baseline/399.tar", "pretrained_source": "shared"},
  "UCM": {"data_dir": "/mnt/sdc/wzj/datasets_causal_splits/ablation_source_splits/UCM_source", "checkpoint": "/mnt/sdc/wzj/SGA-Net-ablation-v2/output/checkpoints/baseline/399.tar", "pretrained_source": "shared"}
}
```

These are placeholders, NOT existing server paths. Each data_dir contains the
source base/val JSON and three target novel JSON. Source val and target novel
require at least 20 images/class (5 support +15 query), at least five classes.
The runner rejects empty splits, missing images and path overlap.  Reusing the
same warmup is rejected by default; because this experiment intentionally uses
one common warmup, every run must explicitly pass
`--allow-shared-checkpoint`, and the manifest records that exception.
It cannot infer pretraining provenance or detect duplicate image content from
different paths; manually verify provenance and data splits. Only trusted torch
checkpoints may be used. The original server splits with empty AID/UCM base/val
and empty NWPU novel are NOT sufficient for nine directions.

`--verify-images` is intentional: it stops before writing a formal split if a
file is truncated or unreadable.  Repair any reported file first; do not silently
drop it or duplicate another image.

```bash
cd /mnt/sdc/wzj/SGA-Net-ablation-v2
git pull --ff-only origin fix/source-only-ablation
export PATH=/mnt/sdc/wzj/envs/sganet/bin:$PATH
python scripts/run_ablation_nine.py --config /path/to/ablation_sources.json --allow-shared-checkpoint --check-only
# After all checks pass: short end-to-end test first, then full jobs.
python scripts/run_ablation_nine.py --config /path/to/ablation_sources.json --allow-shared-checkpoint --gpu 6 --smoke
python scripts/run_ablation_nine.py --config /path/to/ablation_sources.json --allow-shared-checkpoint --gpu 6
```

`--dry-run` prints commands without training or requiring data files.
`--sources NWPU` permits a verified subset. `--arms B C D` excludes A.
Before comparing the existing E results, verify identical splits, warmup,
training budget, evaluation episodes and implementation version. Do not silently
merge an older E produced before the scale-gradient fix into a matched ablation.
Fresh timestamped outputs never overwrite experiments. An error stops execution;
there is no automatic resume. `logs/ablation_nine/<tag>/results.csv` stores per-domain
results; manifest stores command lines, checkpoint and split hashes, commit/status.
Episode intervals are not multi-training-seed confidence intervals.
