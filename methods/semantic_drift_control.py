"""Experimental primitives for semantic-drift-controlled style attacks.

This module contains the reusable primitives used by the semantic-controlled
StyleAdv path. ``SemanticDriftControlConfig.enabled`` defaults to ``False`` so
the existing baseline remains numerically untouched until the new protocol is
enabled explicitly.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class SemanticDriftControlConfig:
    """Configuration for the isolated semantic-drift control primitives."""

    enabled: bool = False
    text_temperature: float = 0.07
    semantic_risk_beta: float = 1.0
    budget_temperature: float = 1.0
    stability_delta: float = 1e-8
    drift_threshold: float = 0.0
    dual_step_size: float = 0.1
    dual_initial_value: float = 0.0
    dual_max_value: Optional[float] = None
    sigma_min: float = 1e-6

    def __post_init__(self) -> None:
        if self.text_temperature <= 0:
            raise ValueError("text_temperature must be positive")
        if self.semantic_risk_beta < 0:
            raise ValueError("semantic_risk_beta must be non-negative")
        if self.budget_temperature <= 0:
            raise ValueError("budget_temperature must be positive")
        if self.stability_delta <= 0:
            raise ValueError("stability_delta must be positive")
        if self.drift_threshold < 0:
            raise ValueError("drift_threshold must be non-negative")
        if self.dual_step_size <= 0:
            raise ValueError("dual_step_size must be positive")
        if self.dual_initial_value < 0:
            raise ValueError("dual_initial_value must be non-negative")
        if self.dual_max_value is not None:
            if self.dual_max_value <= 0:
                raise ValueError("dual_max_value must be positive when provided")
            if self.dual_initial_value > self.dual_max_value:
                raise ValueError("dual_initial_value cannot exceed dual_max_value")
        if self.sigma_min <= 0:
            raise ValueError("sigma_min must be positive")


class TextGroundedSemanticAnchor(nn.Module):
    """Map final visual features into an episode-local frozen text space.

    The caller must provide text prototypes in the exact episode class order.
    They are detached inside ``forward`` because the research boundary requires
    a frozen text encoder.  During an attack inner loop, setting
    ``freeze_projector_for_attack=True`` freezes the projector parameters while
    preserving gradients from the text loss to the visual/style variables.
    The projector can be trained separately by calling the outer path with the
    flag disabled.
    """

    def __init__(self, visual_dim: int, text_dim: int, temperature: float = 0.07):
        super().__init__()
        if visual_dim <= 0 or text_dim <= 0:
            raise ValueError("visual_dim and text_dim must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.visual_dim = int(visual_dim)
        self.text_dim = int(text_dim)
        self.temperature = float(temperature)
        self.projector = nn.Linear(self.visual_dim, self.text_dim)

    def _project(
        self,
        final_features: torch.Tensor,
        freeze_projector_for_attack: bool,
    ) -> torch.Tensor:
        if freeze_projector_for_attack:
            weight = self.projector.weight.detach()
            bias = None if self.projector.bias is None else self.projector.bias.detach()
            return F.linear(final_features, weight, bias)
        return self.projector(final_features)

    def forward(
        self,
        final_features: torch.Tensor,
        episode_text_prototypes: torch.Tensor,
        episode_targets: torch.Tensor,
        *,
        freeze_projector_for_attack: bool = False,
        reduction: str = "mean",
    ) -> torch.Tensor:
        if final_features.ndim != 2:
            raise ValueError(
                "final_features must have shape [N, visual_dim]; pool spatial features explicitly"
            )
        if episode_text_prototypes.ndim != 2:
            raise ValueError("episode_text_prototypes must have shape [K, text_dim]")
        if episode_targets.ndim != 1:
            raise ValueError("episode_targets must have shape [N]")
        if final_features.shape[0] != episode_targets.shape[0]:
            raise ValueError("feature and target batch sizes do not match")
        if final_features.shape[1] != self.visual_dim:
            raise ValueError(
                f"expected visual_dim={self.visual_dim}, got {final_features.shape[1]}"
            )
        if episode_text_prototypes.shape[1] != self.text_dim:
            raise ValueError(
                f"expected text_dim={self.text_dim}, got {episode_text_prototypes.shape[1]}"
            )
        if final_features.device != episode_text_prototypes.device:
            raise ValueError("visual features and text prototypes must share a device")
        if final_features.device != episode_targets.device:
            raise ValueError("visual features and episode targets must share a device")
        if episode_targets.dtype != torch.long:
            raise TypeError("episode_targets must be torch.long episode-local indices")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be one of: none, mean, sum")

        num_classes = episode_text_prototypes.shape[0]
        if num_classes <= 1:
            raise ValueError("an episode must contain at least two text prototypes")
        if episode_targets.numel() > 0:
            target_min = int(episode_targets.min().item())
            target_max = int(episode_targets.max().item())
            if target_min < 0 or target_max >= num_classes:
                raise ValueError(
                    "episode_targets must be local indices in [0, K); map global labels first"
                )

        visual_embeddings = F.normalize(
            self._project(final_features, freeze_projector_for_attack), dim=-1
        )
        text_prototypes = F.normalize(episode_text_prototypes.detach(), dim=-1)
        logits = visual_embeddings @ text_prototypes.transpose(0, 1)
        logits = logits / self.temperature
        return F.cross_entropy(logits, episode_targets, reduction=reduction)


def semantic_drift(
    attacked_text_loss: torch.Tensor,
    clean_text_loss: torch.Tensor,
    *,
    detach_clean: bool = True,
) -> torch.Tensor:
    """Return attacked-minus-clean semantic loss under a shared anchor."""

    clean_reference = clean_text_loss.detach() if detach_clean else clean_text_loss
    return attacked_text_loss - clean_reference


def fixed_budget_channel_weights(
    task_gradient: torch.Tensor,
    semantic_gradient: torch.Tensor,
    *,
    beta: float,
    temperature: float,
    delta: float,
    channel_dim: int = 1,
    detach: bool = True,
) -> torch.Tensor:
    """Compute risk-aware weights whose channel sum equals the channel count.

    Call this function separately for ``mu`` and ``sigma``.  Concatenating them
    would silently couple their budgets and would no longer test the proposed
    per-statistic fixed-budget constraint.
    """

    if task_gradient.shape != semantic_gradient.shape:
        raise ValueError("task and semantic gradients must have identical shapes")
    if task_gradient.ndim == 0:
        raise ValueError("style gradients must include a channel dimension")
    if not task_gradient.is_floating_point() or not semantic_gradient.is_floating_point():
        raise TypeError("style gradients must be floating-point tensors")
    if task_gradient.device != semantic_gradient.device:
        raise ValueError("task and semantic gradients must share a device")
    if beta < 0:
        raise ValueError("beta must be non-negative")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if delta <= 0:
        raise ValueError("delta must be positive")

    normalized_channel_dim = channel_dim % task_gradient.ndim
    channel_count = task_gradient.shape[normalized_channel_dim]
    if channel_count <= 0:
        raise ValueError("channel dimension cannot be empty")
    if not torch.isfinite(task_gradient).all() or not torch.isfinite(semantic_gradient).all():
        raise ValueError("style gradients must be finite")

    calculation_dtype = (
        torch.float32
        if task_gradient.dtype in {torch.float16, torch.bfloat16}
        else task_gradient.dtype
    )
    task_magnitude = task_gradient.abs().to(calculation_dtype)
    semantic_magnitude = semantic_gradient.abs().to(calculation_dtype)
    utility = torch.log(task_magnitude + delta)
    utility = utility - beta * torch.log(semantic_magnitude + delta)
    weights = torch.softmax(utility / temperature, dim=normalized_channel_dim)
    weights = weights * float(channel_count)
    weights = weights.to(task_gradient.dtype)
    return weights.detach() if detach else weights


def weighted_sign_step(
    style: torch.Tensor,
    objective_gradient: torch.Tensor,
    weights: torch.Tensor,
    *,
    step_size: float,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    detach_direction: bool = True,
) -> torch.Tensor:
    """Apply one weighted sign step, optionally clamping validity bounds.

    This is not a projection onto the unresolved attack set ``B_l``.  A caller
    comparing against StyleAdv must add an explicitly selected, logged budget
    projection and must report the realized displacement after any clamp.
    """

    if style.shape != objective_gradient.shape or style.shape != weights.shape:
        raise ValueError("style, objective_gradient, and weights must share a shape")
    if step_size < 0:
        raise ValueError("step_size must be non-negative")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError("minimum cannot exceed maximum")
    if not torch.isfinite(style).all():
        raise ValueError("style must be finite")
    if not torch.isfinite(objective_gradient).all() or not torch.isfinite(weights).all():
        raise ValueError("objective gradients and weights must be finite")

    direction_source = objective_gradient.detach() if detach_direction else objective_gradient
    attacked_style = style + float(step_size) * weights * direction_source.sign()
    if minimum is not None or maximum is not None:
        attacked_style = torch.clamp(attacked_style, min=minimum, max=maximum)
    return attacked_style


def project_style_budget(
    candidate_style: torch.Tensor,
    base_style: torch.Tensor,
    epsilon: float,
    *,
    channel_dim: int = 1,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    delta: float = 1e-8,
) -> Tuple[torch.Tensor, dict]:
    """Project a style candidate onto a per-sample channel-mean L1 budget.

    The budget is measured against ``base_style`` for the current progressive
    stage.  Validity clipping is applied before the final displacement is
    measured; therefore the returned statistics describe the realized attack,
    not the unclipped proposal.
    """

    if candidate_style.shape != base_style.shape:
        raise ValueError("candidate_style and base_style must share a shape")
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if delta <= 0:
        raise ValueError("delta must be positive")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError("minimum cannot exceed maximum")
    if not torch.isfinite(candidate_style).all() or not torch.isfinite(base_style).all():
        raise ValueError("style tensors must be finite")

    normalized_channel_dim = channel_dim % candidate_style.ndim
    unclipped_candidate = candidate_style
    if minimum is not None or maximum is not None:
        clipped_candidate = torch.clamp(candidate_style, min=minimum, max=maximum)
    else:
        clipped_candidate = candidate_style

    displacement = clipped_candidate - base_style
    mean_abs = displacement.abs().mean(dim=normalized_channel_dim, keepdim=True)
    scale = torch.where(
        mean_abs > float(epsilon),
        float(epsilon) / (mean_abs + float(delta)),
        torch.ones_like(mean_abs),
    )
    projected_style = base_style + displacement * scale
    if minimum is not None or maximum is not None:
        projected_style = torch.clamp(projected_style, min=minimum, max=maximum)

    realized_delta = projected_style - base_style
    realized_mean_abs = realized_delta.abs().mean(dim=normalized_channel_dim)
    realized_linf = realized_delta.abs().amax(dim=normalized_channel_dim)
    clamp_rate = realized_delta.new_zeros(())
    if minimum is not None:
        clamp_rate = (unclipped_candidate < float(minimum)).to(realized_delta.dtype).mean()

    stats = {
        "mean_abs_delta": realized_mean_abs.mean().detach(),
        "linf_delta": realized_linf.mean().detach(),
        "budget_utilization": (
            realized_mean_abs.mean() / max(float(epsilon), float(delta))
        ).detach(),
        "sigma_clamp_rate": clamp_rate.detach(),
    }
    return projected_style, stats


def controlled_style_step(
    style: torch.Tensor,
    task_gradient: torch.Tensor,
    semantic_gradient: torch.Tensor,
    objective_gradient: torch.Tensor,
    *,
    step_size: float,
    config: SemanticDriftControlConfig,
    channel_dim: int = 1,
    is_sigma: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Apply the fixed-budget primitive, or return the baseline input unchanged."""

    if not config.enabled:
        return style, None
    weights = fixed_budget_channel_weights(
        task_gradient,
        semantic_gradient,
        beta=config.semantic_risk_beta,
        temperature=config.budget_temperature,
        delta=config.stability_delta,
        channel_dim=channel_dim,
        detach=True,
    )
    attacked_style = weighted_sign_step(
        style,
        objective_gradient,
        weights,
        step_size=step_size,
        minimum=config.sigma_min if is_sigma else None,
        detach_direction=True,
    )
    return attacked_style, weights


def initial_dual_state(
    reference: torch.Tensor,
    config: SemanticDriftControlConfig,
) -> torch.Tensor:
    """Create a detached scalar lambda state on the reference device/dtype."""

    if not reference.is_floating_point():
        raise TypeError("reference must be floating point")
    return reference.new_tensor(config.dual_initial_value).detach()


def stage_lagrangian(
    task_loss: torch.Tensor,
    drift: torch.Tensor,
    dual_state: torch.Tensor,
    config: SemanticDriftControlConfig,
) -> torch.Tensor:
    """Compute ``L_task - lambda * (drift - tau)`` for one depth stage."""

    if not config.enabled:
        return task_loss
    if task_loss.numel() != 1 or drift.numel() != 1 or dual_state.numel() != 1:
        raise ValueError("task_loss, drift, and dual_state must be scalar tensors")
    if task_loss.device != drift.device or task_loss.device != dual_state.device:
        raise ValueError("task_loss, drift, and dual_state must share a device")
    return task_loss - dual_state.detach() * (drift - config.drift_threshold)


@torch.no_grad()
def update_dual_state(
    dual_state: torch.Tensor,
    drift: torch.Tensor,
    config: SemanticDriftControlConfig,
) -> torch.Tensor:
    """Update lambda from measured drift and stop gradients across stages."""

    if not config.enabled:
        return dual_state.detach()
    if dual_state.numel() != 1 or drift.numel() != 1:
        raise ValueError("dual_state and drift must be scalar tensors")
    if dual_state.device != drift.device:
        raise ValueError("dual_state and drift must share a device")
    next_state = dual_state.detach() + config.dual_step_size * (
        drift.detach() - config.drift_threshold
    )
    next_state = next_state.clamp_min(0.0)
    if config.dual_max_value is not None:
        next_state = next_state.clamp_max(config.dual_max_value)
    return next_state.detach()


__all__ = [
    "SemanticDriftControlConfig",
    "TextGroundedSemanticAnchor",
    "semantic_drift",
    "fixed_budget_channel_weights",
    "weighted_sign_step",
    "project_style_budget",
    "controlled_style_step",
    "initial_dual_state",
    "stage_lagrangian",
    "update_dual_state",
]
