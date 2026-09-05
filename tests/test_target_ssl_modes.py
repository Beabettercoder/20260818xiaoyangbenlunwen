import unittest

import torch

from methods.StyleAdv_RN_GNN import StyleAdvGNN


class _TargetSSLHarness:
  compute_target_unlabeled_loss = StyleAdvGNN.compute_target_unlabeled_loss
  adjust_target_ssl_weight = StyleAdvGNN.adjust_target_ssl_weight

  def __init__(self, mode):
    self.target_ssl_mode = mode
    self.target_ssl_confidence_threshold = 0.8
    self.target_ssl_weight_floor = 0.0
    self.target_ssl_feature_weight = 0.25
    self.target_ssl_prototype_weight = 0.25

  def forward_global_features(self, x):
    return x


class TargetSSLModeTests(unittest.TestCase):
  def test_feature_consistency_is_label_free_and_backpropagates_to_strong_view(self):
    harness = _TargetSSLHarness("feature_consistency")
    weak = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    strong = torch.tensor([[0.0, 1.0], [1.0, 0.0]], requires_grad=True)

    loss, stats = harness.compute_target_unlabeled_loss(weak, strong)
    loss.backward()

    self.assertAlmostEqual(float(loss.detach()), 1.0, places=6)
    self.assertIsNone(weak.grad)
    self.assertIsNotNone(strong.grad)
    self.assertEqual(stats["selected_ratio"], 1.0)
    self.assertEqual(stats["pseudo_confidence"], 0.0)

  def test_feature_consistency_keeps_requested_ramp_weight(self):
    harness = _TargetSSLHarness("feature_consistency")
    weight, factor = harness.adjust_target_ssl_weight(0.25, {"selected_ratio": 1.0})
    self.assertEqual(weight, 0.25)
    self.assertEqual(factor, 1.0)


if __name__ == "__main__":
  unittest.main()
