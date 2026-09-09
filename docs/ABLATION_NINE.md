# Nine source-only transfers

Entry point: `scripts/run_ablation_nine.py`. GPU defaults to **5**, sequential.
Sources NWPU/AID/UCM each test the other three datasets (including EuroSAT).
Only A/B/C/D are scheduled; E is already available and is not rerun or merged.
Both shots: 18 training jobs, 72 evaluations. No target data enters training.

A directly evaluates the source-pretrained ResNet10 with cosine prototypes;
it is NOT random initialization and receives no further training. B independent
attack; C progressive attack; D independent + text scales; E progressive + text
scales. B–E load the corresponding source checkpoint and use the existing GNN.
Therefore A–B changes training budget AND prediction head, not just attack.
Use B/C/D/E for component comparisons. No best-score ordering is guaranteed.
Existing GNN balanced-query grouping is unchanged; this script does not resolve
that evaluation-protocol limitation. Do not describe it as label-independent grouping.

Create a JSON config with three entries, each using **verified real paths**:

```json
{
  "NWPU": {"data_dir": "/path/to/NWPU_source_splits", "checkpoint": "/path/to/NWPU/399.tar", "pretrained_source": "NWPU"},
  "AID": {"data_dir": "/path/to/AID_source_splits", "checkpoint": "/path/to/AID/399.tar", "pretrained_source": "AID"},
  "UCM": {"data_dir": "/path/to/UCM_source_splits", "checkpoint": "/path/to/UCM/399.tar", "pretrained_source": "UCM"}
}
```

These are placeholders, NOT existing server paths. Each data_dir contains the
source base/val JSON and three target novel JSON. Source val and target novel
require at least 20 images/class (5 support +15 query), at least five classes.
The runner rejects empty splits, missing images, path overlap and reused weights.
It cannot infer pretraining provenance or detect duplicate image content from
different paths; manually verify provenance and data splits. Only trusted torch
checkpoints may be used. The original server splits with empty AID/UCM base/val
and empty NWPU novel are NOT sufficient for nine directions.

```bash
cd /mnt/sdc/wzj/SGA-Net-ablation-v2
git pull --ff-only origin fix/source-only-ablation
export PATH=/mnt/sdc/wzj/envs/sganet/bin:$PATH
python scripts/run_ablation_nine.py --config /path/to/ablation_sources.json --check-only
# After all checks pass: short end-to-end test first, then full jobs.
python scripts/run_ablation_nine.py --config /path/to/ablation_sources.json --gpu 5 --smoke
python scripts/run_ablation_nine.py --config /path/to/ablation_sources.json --gpu 5
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
