# Target-unlabeled trial protocol

This trial addresses a failure in the legacy target SSL path. Its pseudo-labels
are predictions from the 64-way NWPU source classifier. Under the observed
NWPU-to-AID run, the mean confidence was about 0.024 and the default filter
selected no target samples. Removing the filter selected all samples but made
the source-class pseudo-label objective unreliable.

The new `feature_consistency` mode uses only weak/strong target views:

`L_target = mean(1 - cosine(stopgrad(f(x_weak)), f(x_strong)))`.

It does not infer target class labels and does not use target query labels. The
source episodic objective remains active and anchors the representation.

`scripts/run_target_unlabeled_trial.sh` runs two matched arms:

1. `pairwise_control`: NWPU base+val training with target SSL weight zero.
2. `target_feature`: the same setup plus target feature consistency.

The default is a short diagnostic (AID, 5-shot, one epoch). It is not a paper
result. Promote it to a formal run only after the target feature loss is finite
and nonzero, both arms finish, and the protocol summary is retained.
