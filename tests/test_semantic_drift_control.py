import unittest

import torch
import torch.nn.functional as F

from methods.semantic_drift_control import (
    SemanticDriftControlConfig,
    TextGroundedSemanticAnchor,
    controlled_style_step,
    fixed_budget_channel_weights,
    initial_dual_state,
    project_style_budget,
    semantic_drift,
    stage_lagrangian,
    update_dual_state,
)


class SemanticDriftControlSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    @staticmethod
    def _identity_anchor(dim: int = 3) -> TextGroundedSemanticAnchor:
        anchor = TextGroundedSemanticAnchor(dim, dim, temperature=0.2)
        with torch.no_grad():
            anchor.projector.weight.copy_(torch.eye(dim))
            anchor.projector.bias.zero_()
        return anchor

    def test_correct_and_shuffled_text_take_different_semantic_branches(self) -> None:
        anchor = self._identity_anchor()
        features = torch.eye(3)
        text = torch.eye(3)
        targets = torch.arange(3, dtype=torch.long)

        correct_loss = anchor(
            features,
            text,
            targets,
            freeze_projector_for_attack=True,
        )
        shuffled_loss = anchor(
            features,
            text[[1, 2, 0]],
            targets,
            freeze_projector_for_attack=True,
        )

        self.assertLess(correct_loss.item(), shuffled_loss.item())
        self.assertGreater((shuffled_loss - correct_loss).item(), 1.0)

    def test_anchor_requires_episode_local_target_indices(self) -> None:
        anchor = self._identity_anchor(dim=2)
        with self.assertRaisesRegex(ValueError, "local indices"):
            anchor(
                torch.eye(2),
                torch.eye(2),
                torch.tensor([10, 11], dtype=torch.long),
            )

    def test_style_gradient_reaches_attack_variable_but_not_frozen_components(self) -> None:
        anchor = self._identity_anchor()
        style_variable = torch.tensor(
            [[0.8, 0.2, 0.1], [0.1, 0.7, 0.2], [0.2, 0.1, 0.8]],
            requires_grad=True,
        )
        frozen_text = torch.nn.Parameter(torch.eye(3), requires_grad=False)
        targets = torch.arange(3, dtype=torch.long)

        loss = anchor(
            style_variable,
            frozen_text,
            targets,
            freeze_projector_for_attack=True,
        )
        loss.backward()

        self.assertIsNotNone(style_variable.grad)
        self.assertGreater(style_variable.grad.abs().sum().item(), 0.0)
        self.assertIsNone(anchor.projector.weight.grad)
        self.assertIsNone(anchor.projector.bias.grad)
        self.assertIsNone(frozen_text.grad)

    def test_outer_path_updates_projector_but_not_detached_visual_features(self) -> None:
        anchor = TextGroundedSemanticAnchor(visual_dim=4, text_dim=3, temperature=0.2)
        visual_features = torch.randn(6, 4, requires_grad=True)
        text_prototypes = F.normalize(torch.randn(3, 3), dim=-1)
        targets = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)

        loss = anchor(
            visual_features.detach(),
            text_prototypes,
            targets,
            freeze_projector_for_attack=False,
        )
        loss.backward()

        self.assertIsNone(visual_features.grad)
        self.assertIsNotNone(anchor.projector.weight.grad)
        self.assertIsNotNone(anchor.projector.bias.grad)
        self.assertGreater(anchor.projector.weight.grad.abs().sum().item(), 0.0)

    def test_mu_and_sigma_each_receive_an_independent_fixed_budget(self) -> None:
        mu_task = torch.randn(2, 4, 1, 1)
        mu_risk = torch.randn(2, 4, 1, 1)
        sigma_task = torch.randn(2, 4, 1, 1)
        sigma_risk = torch.randn(2, 4, 1, 1)

        mu_weights = fixed_budget_channel_weights(
            mu_task,
            mu_risk,
            beta=0.7,
            temperature=0.8,
            delta=1e-8,
            detach=True,
        )
        sigma_weights = fixed_budget_channel_weights(
            sigma_task,
            sigma_risk,
            beta=0.7,
            temperature=0.8,
            delta=1e-8,
            detach=True,
        )

        expected = torch.full((2, 1, 1, 1), 4.0)
        self.assertTrue(torch.allclose(mu_weights.sum(dim=1, keepdim=True), expected))
        self.assertTrue(torch.allclose(sigma_weights.sum(dim=1, keepdim=True), expected))

    def test_detached_weights_do_not_create_a_second_order_path(self) -> None:
        style = torch.tensor([[[[0.3]], [[0.5]], [[0.7]]]], requires_grad=True)
        task_loss = style.square().sum()
        semantic_loss = style.pow(3).sum()
        task_gradient = torch.autograd.grad(
            task_loss, style, create_graph=True, retain_graph=True
        )[0]
        semantic_gradient = torch.autograd.grad(
            semantic_loss, style, create_graph=True, retain_graph=True
        )[0]

        weights = fixed_budget_channel_weights(
            task_gradient,
            semantic_gradient,
            beta=1.0,
            temperature=1.0,
            delta=1e-8,
            detach=True,
        )

        self.assertFalse(weights.requires_grad)
        self.assertIsNone(weights.grad_fn)
        config = SemanticDriftControlConfig(enabled=True)
        attacked, returned_weights = controlled_style_step(
            style,
            task_gradient,
            semantic_gradient,
            task_gradient - semantic_gradient,
            step_size=0.1,
            config=config,
        )
        attacked.sum().backward()
        self.assertIsNotNone(style.grad)
        self.assertFalse(returned_weights.requires_grad)

    def test_sigma_remains_valid_after_a_negative_step(self) -> None:
        sigma = torch.full((1, 3, 1, 1), 0.01)
        task_gradient = torch.ones_like(sigma)
        semantic_gradient = torch.zeros_like(sigma)
        objective_gradient = -torch.ones_like(sigma)
        config = SemanticDriftControlConfig(enabled=True, sigma_min=1e-4)

        attacked_sigma, _ = controlled_style_step(
            sigma,
            task_gradient,
            semantic_gradient,
            objective_gradient,
            step_size=1.0,
            config=config,
            is_sigma=True,
        )

        self.assertTrue(torch.all(attacked_sigma >= config.sigma_min))
        self.assertTrue(torch.isfinite(attacked_sigma).all())

    def test_projected_style_budget_reports_real_displacement_after_sigma_clamp(self) -> None:
        base = torch.full((2, 4, 1, 1), 0.5)
        candidate = torch.tensor(
            [
                [[[0.0]], [[0.0]], [[0.0]], [[0.0]]],
                [[[1.5]], [[1.5]], [[1.5]], [[1.5]]],
            ]
        )
        projected, stats = project_style_budget(
            candidate,
            base,
            epsilon=0.1,
            minimum=0.2,
        )

        realized = (projected - base).abs().mean(dim=1)
        self.assertTrue(torch.all(realized <= 0.1 + 1e-6))
        self.assertTrue(torch.all(projected >= 0.2))
        self.assertGreater(stats["sigma_clamp_rate"].item(), 0.0)
        self.assertLessEqual(stats["budget_utilization"].item(), 1.0 + 1e-6)

    def test_feedback_lambda_tracks_drift_and_changes_next_stage_gradient(self) -> None:
        config = SemanticDriftControlConfig(
            enabled=True,
            drift_threshold=0.2,
            dual_step_size=0.5,
            dual_initial_value=0.4,
        )
        reference = torch.tensor(0.0)
        state = initial_dual_state(reference, config)
        increased = update_dual_state(state, torch.tensor(0.6), config)
        decreased = update_dual_state(state, torch.tensor(0.0), config)

        self.assertAlmostEqual(increased.item(), 0.6, places=6)
        self.assertAlmostEqual(decreased.item(), 0.3, places=6)

        next_style_a = torch.tensor(1.0, requires_grad=True)
        objective_a = stage_lagrangian(
            3.0 * next_style_a,
            next_style_a,
            state,
            config,
        )
        gradient_a = torch.autograd.grad(objective_a, next_style_a)[0]

        next_style_b = torch.tensor(1.0, requires_grad=True)
        objective_b = stage_lagrangian(
            3.0 * next_style_b,
            next_style_b,
            increased,
            config,
        )
        gradient_b = torch.autograd.grad(objective_b, next_style_b)[0]

        self.assertNotEqual(gradient_a.item(), gradient_b.item())
        self.assertLess(gradient_b.item(), gradient_a.item())

    def test_clean_reference_is_detached_in_drift(self) -> None:
        clean = torch.tensor(0.2, requires_grad=True)
        attacked = torch.tensor(0.7, requires_grad=True)
        drift = semantic_drift(attacked, clean)
        drift.backward()
        self.assertEqual(attacked.grad.item(), 1.0)
        self.assertIsNone(clean.grad)

    def test_disabled_switch_is_an_exact_identity(self) -> None:
        config = SemanticDriftControlConfig(enabled=False)
        style = torch.randn(1, 3, 1, 1)
        gradient = torch.randn_like(style)
        unchanged, weights = controlled_style_step(
            style,
            gradient,
            gradient,
            gradient,
            step_size=100.0,
            config=config,
        )
        task_loss = torch.tensor(2.0, requires_grad=True)
        objective = stage_lagrangian(
            task_loss,
            torch.tensor(999.0),
            torch.tensor(3.0),
            config,
        )

        self.assertIs(unchanged, style)
        self.assertIsNone(weights)
        self.assertIs(objective, task_loss)


if __name__ == "__main__":
    unittest.main()
