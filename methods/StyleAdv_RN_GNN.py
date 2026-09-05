import math
import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import Optional, Tuple, List

from methods.gnn import GNN_nl
from methods import backbone_multiblock
from methods.tool_func import *
from methods.meta_template_StyleAdv_RN_GNN import MetaTemplate
from methods.semantic_drift_control import (
  SemanticDriftControlConfig,
  TextGroundedSemanticAnchor,
  fixed_budget_channel_weights,
  project_style_budget,
  update_dual_state,
  weighted_sign_step,
)
from utils.distributed_utils import is_main_process

# 灏濊瘯瀵煎叆鏂囨湰寮曞鐩稿叧妯″潡
try:
  from methods.style_prompt_modules import TextPromptEncoder, StylePromptGenerator, ChannelGatingNetwork
  from methods.text_prompt import load_class_names, default_template, get_ensemble_templates
  CLIP_AVAILABLE = True
except ImportError:
  CLIP_AVAILABLE = False


class StyleAdvGNN(MetaTemplate):
  maml=False
  def __init__(
    self,
    model_func,
    n_way,
    n_support,
    tf_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    # ============ 鏂板: 鏂囨湰寮曞鍙傛暟 ============
    dataset_name: Optional[str] = None,
    data_root: Optional[str] = None,
    text_guide_epsilon: int = 0,      # 鏂囨湰寮曞epsilon缂╂斁
    text_guide_gradient: int = 0,     # 鏂囨湰寮曞姊害鏂瑰悜
    clip_model_name: str = "ViT-B/32",
    text_weight_mode: str = "adaptive",
    text_weight_k: float = 0.2,
    text_weight_fixed: float = 0.5,
    text_confidence_floor: float = 0.35,
    gate_min: float = 0.0,
    gate_max: float = 1.0,
    epsilon_scale_min: float = 0.5,
    epsilon_scale_max: float = 1.5,
    epsilon_block1: float = 0.8,
    epsilon_block2: float = 0.08,
    epsilon_block3: float = 0.008,
    style_attack_mode: str = "progressive",
    use_prompt_ensemble: bool = True,
    text_calibration_weight: float = 0.0,
    semantic_anchor: int = 0,
    semantic_anchor_weight: float = 1.0,
    semantic_anchor_temperature: float = 0.07,
    semantic_drift_control: int = 0,
    semantic_risk_beta: float = 1.0,
    semantic_budget_temperature: float = 1.0,
    semantic_drift_threshold: float = 0.0,
    semantic_dual_step_size: float = 0.1,
    semantic_lambda_max: float = 10.0,
    semantic_sigma_min: float = 1e-6,
    target_ssl_temperature: float = 0.5,
    target_ssl_mode: str = "legacy_pseudo",
    target_ssl_confidence_threshold: float = 0.80,
    target_ssl_max_entropy: float = 0.75,
    target_ssl_view_threshold: float = 0.65,
    target_ssl_feature_weight: float = 0.25,
    target_ssl_prototype_weight: float = 0.25,
    target_ssl_weight_floor: float = 0.0,
    eval_episode_relevance: int = 0,
    eval_proto_refine: int = 0,
    eval_proto_refine_alpha_1shot: float = 8.0,
    eval_proto_refine_alpha_5shot: float = 4.0,
    eval_proto_refine_affinity_threshold: float = 0.55,
    eval_proto_refine_margin_threshold: float = 0.05,
    eval_proto_refine_min_selected: int = 8,
    eval_proto_refine_min_coverage: float = 0.40,
    eval_proto_refine_affinity_threshold_1shot: float = 0.70,
    eval_proto_refine_affinity_threshold_5shot: float = 0.65,
    eval_proto_refine_margin_threshold_1shot: float = 0.12,
    eval_proto_refine_margin_threshold_5shot: float = 0.10,
    eval_proto_refine_min_selected_1shot: int = 15,
    eval_proto_refine_min_selected_5shot: int = 12,
    eval_proto_refine_min_coverage_1shot: float = 0.60,
    eval_proto_refine_min_coverage_5shot: float = 0.50,
    eval_proto_refine_topk_1shot: int = 15,
    eval_proto_refine_support_loo_guard: int = 1,
    eval_proto_refine_support_loo_tolerance: float = 0.0,
    eval_proto_refine_support_margin_guard: int = 1,
    eval_proto_refine_support_margin_tolerance: float = 0.0,
    eval_proto_refine_target_guard: int = 1,
    eval_proto_refine_target_affinity_tolerance: float = 0.0,
    eval_proto_refine_target_margin_tolerance: float = 0.0,
    eval_proto_refine_target_affinity_gain_1shot: float = 0.05,
    eval_proto_refine_target_affinity_gain_5shot: float = 0.02,
    eval_proto_refine_target_margin_gain_1shot: float = 0.05,
    eval_proto_refine_target_margin_gain_5shot: float = 0.02,
  ):
    self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    super(StyleAdvGNN, self).__init__(model_func, n_way, n_support, tf_path=tf_path, device=self.device)
    
    # keep n_way fixed to the configured value to avoid GNN shape drift (align with v3)
    self.change_way = False

    # loss function
    self.loss_fn = nn.CrossEntropyLoss()

    # metric function
    self.metric_input_dim = 128
    if not self.maml:
      self.fc = nn.Sequential(
        nn.Linear(self.feat_dim, self.metric_input_dim),
        nn.BatchNorm1d(self.metric_input_dim, track_running_stats=False),
      )
    else:
      self.fc = nn.Sequential(
        backbone_multiblock.Linear_fw(self.feat_dim, self.metric_input_dim),
        backbone_multiblock.BatchNorm1d_fw(self.metric_input_dim, track_running_stats=False),
      )
    
    # global classifier for StyleAdv
    self.method = 'GnnNet'
    self.classifier = nn.Linear(self.feature.final_feat_dim, 64)

    # fixed support labels for GNN input
    support_label = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).long().unsqueeze(1)
    support_label = torch.zeros(self.n_way*self.n_support, self.n_way).scatter(1, support_label, 1).view(self.n_way, self.n_support, self.n_way)
    support_label = torch.cat([support_label, torch.zeros(self.n_way, 1, self.n_way)], dim=1)
    self.register_buffer('support_label', support_label.view(1, -1, self.n_way))
    
    self.metric_dim = self.metric_input_dim
    self.gnn = GNN_nl(self.metric_dim + self.n_way, 96, self.n_way).to(self.device)

    # ============ 鏂板: 鏂囨湰寮曞缁勪欢鍒濆鍖?============
    self.dataset_name = dataset_name
    self.data_root = data_root
    self.text_guide_epsilon = bool(text_guide_epsilon)
    self.text_guide_gradient = bool(text_guide_gradient)
    self.clip_model_name = clip_model_name
    self._warned_missing_prompts = False
    
    # ============ [娑堣瀺瀹為獙] 淇濆瓨瓒呭弬鏁?============
    self.text_weight_mode = text_weight_mode
    self.text_weight_k = text_weight_k
    self.text_weight_fixed = text_weight_fixed
    self.text_confidence_floor = text_confidence_floor
    self.gate_min = gate_min
    self.gate_max = gate_max
    self.epsilon_scale_min = epsilon_scale_min
    self.epsilon_scale_max = epsilon_scale_max
    self.attack_epsilon_list = [epsilon_block1, epsilon_block2, epsilon_block3]
    self.style_attack_mode = style_attack_mode if style_attack_mode in ["progressive", "independent"] else "progressive"
    
    # ============ [Step1: PromptEnsemble] ============
    self.use_prompt_ensemble = use_prompt_ensemble
    self.semantic_anchor_enabled = bool(semantic_anchor)
    self.semantic_anchor_weight = float(semantic_anchor_weight)
    self.semantic_anchor_temperature = float(semantic_anchor_temperature)
    self.semantic_drift_control_enabled = bool(semantic_drift_control)
    if self.semantic_drift_control_enabled and not self.semantic_anchor_enabled:
      raise ValueError('semantic_drift_control requires semantic_anchor=1')
    self.semantic_drift_config = SemanticDriftControlConfig(
      enabled=self.semantic_drift_control_enabled,
      semantic_risk_beta=float(semantic_risk_beta),
      budget_temperature=float(semantic_budget_temperature),
      drift_threshold=float(semantic_drift_threshold),
      dual_step_size=float(semantic_dual_step_size),
      dual_max_value=float(semantic_lambda_max),
      sigma_min=float(semantic_sigma_min),
    )
    if self.semantic_anchor_weight < 0:
      raise ValueError('semantic_anchor_weight must be non-negative')
    if self.semantic_anchor_temperature <= 0:
      raise ValueError('semantic_anchor_temperature must be positive')
    self.semantic_anchor: Optional[nn.Module] = None
    self._semantic_text_cache = {}
    self.last_semantic_anchor_loss = 0.0
    self.last_semantic_attack_stats = {}
    self.disable_style_adv_generator = False
    self.last_text_guidance_weight = 1.0
    self.last_text_prompt_agreement = 1.0
    self.target_ssl_temperature = float(target_ssl_temperature)
    if target_ssl_mode not in {"legacy_pseudo", "feature_consistency"}:
      raise ValueError(f"Unsupported target_ssl_mode: {target_ssl_mode}")
    self.target_ssl_mode = target_ssl_mode
    self.target_ssl_confidence_threshold = float(target_ssl_confidence_threshold)
    self.target_ssl_max_entropy = float(target_ssl_max_entropy)
    self.target_ssl_view_threshold = float(target_ssl_view_threshold)
    self.target_ssl_feature_weight = float(target_ssl_feature_weight)
    self.target_ssl_prototype_weight = float(target_ssl_prototype_weight)
    self.target_ssl_weight_floor = float(target_ssl_weight_floor)
    self.eval_episode_relevance = bool(eval_episode_relevance)
    self.eval_proto_refine = bool(eval_proto_refine)
    self.eval_proto_refine_alpha_1shot = float(eval_proto_refine_alpha_1shot)
    self.eval_proto_refine_alpha_5shot = float(eval_proto_refine_alpha_5shot)
    self.eval_proto_refine_affinity_threshold = float(eval_proto_refine_affinity_threshold)
    self.eval_proto_refine_margin_threshold = float(eval_proto_refine_margin_threshold)
    self.eval_proto_refine_min_selected = int(eval_proto_refine_min_selected)
    self.eval_proto_refine_min_coverage = float(eval_proto_refine_min_coverage)
    self.eval_proto_refine_affinity_threshold_1shot = float(eval_proto_refine_affinity_threshold_1shot)
    self.eval_proto_refine_affinity_threshold_5shot = float(eval_proto_refine_affinity_threshold_5shot)
    self.eval_proto_refine_margin_threshold_1shot = float(eval_proto_refine_margin_threshold_1shot)
    self.eval_proto_refine_margin_threshold_5shot = float(eval_proto_refine_margin_threshold_5shot)
    self.eval_proto_refine_min_selected_1shot = int(eval_proto_refine_min_selected_1shot)
    self.eval_proto_refine_min_selected_5shot = int(eval_proto_refine_min_selected_5shot)
    self.eval_proto_refine_min_coverage_1shot = float(eval_proto_refine_min_coverage_1shot)
    self.eval_proto_refine_min_coverage_5shot = float(eval_proto_refine_min_coverage_5shot)
    self.eval_proto_refine_topk_1shot = int(eval_proto_refine_topk_1shot)
    self.eval_proto_refine_support_loo_guard = bool(eval_proto_refine_support_loo_guard)
    self.eval_proto_refine_support_loo_tolerance = float(eval_proto_refine_support_loo_tolerance)
    self.eval_proto_refine_support_margin_guard = bool(eval_proto_refine_support_margin_guard)
    self.eval_proto_refine_support_margin_tolerance = float(eval_proto_refine_support_margin_tolerance)
    self.eval_proto_refine_target_guard = bool(eval_proto_refine_target_guard)
    self.eval_proto_refine_target_affinity_tolerance = float(eval_proto_refine_target_affinity_tolerance)
    self.eval_proto_refine_target_margin_tolerance = float(eval_proto_refine_target_margin_tolerance)
    self.eval_proto_refine_target_affinity_gain_1shot = float(eval_proto_refine_target_affinity_gain_1shot)
    self.eval_proto_refine_target_affinity_gain_5shot = float(eval_proto_refine_target_affinity_gain_5shot)
    self.eval_proto_refine_target_margin_gain_1shot = float(eval_proto_refine_target_margin_gain_1shot)
    self.eval_proto_refine_target_margin_gain_5shot = float(eval_proto_refine_target_margin_gain_5shot)
    self.last_target_ssl_weight_factor = 1.0
    self.last_target_ssl_keep_ratio = 1.0
    self.last_target_ssl_selected_confidence = 0.0
    self.last_eval_episode_stats = {}
    
    # ============ [Step1.1: Test-time Text Calibration] ============
    self.text_calibration_weight = text_calibration_weight
    self._text_calibration_cache = {}  # 缂撳瓨text embeddings閬垮厤閲嶅璁＄畻
    
    # 鏂囨湰缂栫爜鍣ㄥ拰鐢熸垚鍣?
    self.text_prompt_encoder: Optional[nn.Module] = None
    self.style_prompt_generator: Optional[nn.Module] = None
    self.epsilon_scale_head: Optional[nn.Module] = None      # 鐢ㄤ簬epsilon缂╂斁
    self.gradient_gate_network: Optional[nn.Module] = None   # 鐢ㄤ簬姊害闂ㄦ帶
    
    # 鍒濆鍖栨枃鏈紩瀵肩粍浠?
    # [Step1.1] 濡傛灉鍚敤text_calibration锛屼篃闇€瑕佸垵濮嬪寲鏂囨湰妯″潡
    need_text_modules = (
      self.text_guide_epsilon
      or self.text_guide_gradient
      or self.text_calibration_weight > 0
      or self.semantic_anchor_enabled
    )
    if need_text_modules and CLIP_AVAILABLE:
      self._init_text_guidance_modules()
    elif need_text_modules and not CLIP_AVAILABLE:
      warnings.warn("CLIP not available, disabling text guidance. Install: pip install git+https://github.com/openai/CLIP.git")
      self.text_guide_epsilon = False
      self.text_guide_gradient = False
      self.semantic_anchor_enabled = False
      self.semantic_drift_control_enabled = False
      self.semantic_drift_config = SemanticDriftControlConfig(
        enabled=False,
        text_temperature=self.semantic_drift_config.text_temperature,
        semantic_risk_beta=self.semantic_drift_config.semantic_risk_beta,
        budget_temperature=self.semantic_drift_config.budget_temperature,
        stability_delta=self.semantic_drift_config.stability_delta,
        drift_threshold=self.semantic_drift_config.drift_threshold,
        dual_step_size=self.semantic_drift_config.dual_step_size,
        dual_initial_value=self.semantic_drift_config.dual_initial_value,
        dual_max_value=self.semantic_drift_config.dual_max_value,
        sigma_min=self.semantic_drift_config.sigma_min,
      )

  def _init_text_guidance_modules(self):
    """Initialize text-guidance modules."""
    try:
      if is_main_process():
        print(f"[TG-MSAT] Initializing text guidance modules...")
        print(f"  - text_guide_epsilon: {self.text_guide_epsilon}")
        print(f"  - text_guide_gradient: {self.text_guide_gradient}")
        print(f"  - text_weight_mode: {self.text_weight_mode}, k={self.text_weight_k}, fixed={self.text_weight_fixed}")
        print(f"  - text_confidence_floor: {self.text_confidence_floor}")
        print(f"  - gate_range: [{self.gate_min}, {self.gate_max}]")
        print(f"  - epsilon_scale_range: [{self.epsilon_scale_min}, {self.epsilon_scale_max}]")
        print(f"  - attack_eps: {self.attack_epsilon_list}")
        print(f"  - style_attack_mode: {self.style_attack_mode}")
        print(f"  - [Step1] use_prompt_ensemble: {self.use_prompt_ensemble}")
        print(f"  - [Step1.1] text_calibration_weight: {self.text_calibration_weight}")
        print(
          "  - [TargetSSL] "
          f"mode={self.target_ssl_mode}, "
          f"thr={self.target_ssl_confidence_threshold}, "
          f"max_entropy={self.target_ssl_max_entropy}, "
          f"view_thr={self.target_ssl_view_threshold}, "
          f"feat_w={self.target_ssl_feature_weight}, "
          f"proto_w={self.target_ssl_prototype_weight}, "
          f"weight_floor={self.target_ssl_weight_floor}"
        )
        print(
          "  - [EvalProto] "
          f"alpha_1={self.eval_proto_refine_alpha_1shot}, "
          f"alpha_5={self.eval_proto_refine_alpha_5shot}, "
          f"aff_1={self.eval_proto_refine_affinity_threshold_1shot}, "
          f"aff_5={self.eval_proto_refine_affinity_threshold_5shot}, "
          f"margin_1={self.eval_proto_refine_margin_threshold_1shot}, "
          f"margin_5={self.eval_proto_refine_margin_threshold_5shot}, "
          f"topk_1={self.eval_proto_refine_topk_1shot}"
        )
        print(
          "  - [EvalGuard] "
          f"target={self.eval_proto_refine_target_guard}, "
          f"target_aff_tol={self.eval_proto_refine_target_affinity_tolerance}, "
          f"target_margin_tol={self.eval_proto_refine_target_margin_tolerance}, "
          f"target_aff_gain_1={self.eval_proto_refine_target_affinity_gain_1shot}, "
          f"target_aff_gain_5={self.eval_proto_refine_target_affinity_gain_5shot}, "
          f"target_margin_gain_1={self.eval_proto_refine_target_margin_gain_1shot}, "
          f"target_margin_gain_5={self.eval_proto_refine_target_margin_gain_5shot}, "
          f"loo={self.eval_proto_refine_support_loo_guard}, "
          f"loo_tol={self.eval_proto_refine_support_loo_tolerance}, "
          f"margin={self.eval_proto_refine_support_margin_guard}, "
          f"margin_tol={self.eval_proto_refine_support_margin_tolerance}"
        )
      
      # 1. 鏂囨湰缂栫爜鍣?(CLIP)
      self.text_prompt_encoder = TextPromptEncoder(
        clip_model_name=self.clip_model_name, 
        device=self.device
      )
      text_dim = self.text_prompt_encoder.text_dim  # 閫氬父鏄?12
      
      # 2. Legacy style-prompt path.  The semantic anchor does not depend on it.
      style_prompt_dim = 256
      if self.text_guide_epsilon or self.text_guide_gradient:
        self.style_prompt_generator = StylePromptGenerator(
          in_dim=text_dim,
          out_dim=style_prompt_dim,
          hidden_dim=384
        )
      
      # 3. Epsilon缂╂斁澶? style_prompt -> 3涓猙lock鐨別psilon缂╂斁鍥犲瓙
      if self.text_guide_epsilon:
        self.epsilon_scale_head = nn.Sequential(
          nn.Linear(style_prompt_dim, 64),
          nn.ReLU(inplace=True),
          nn.Linear(64, 3),
          nn.Sigmoid()  # 杈撳嚭 [0, 1], 瀹為檯缂╂斁鍒?[epsilon_scale_min, epsilon_scale_max]
        )
        if is_main_process():
          print(f"  - Epsilon scaling head initialized")
      
      # 4. 姊害闂ㄦ帶缃戠粶: style_prompt -> 姣忎釜block閫氶亾绾у埆鐨刧ate
      if self.text_guide_gradient:
        # ResNet10鐨刡lock閫氶亾鏁? block1=64, block2=128, block3=256
        block_channels = [64, 128, 256]
        self.gradient_gate_network = ChannelGatingNetwork(
          style_prompt_dim=style_prompt_dim,
          block_channels=block_channels,
          gate_min=self.gate_min,
          gate_max=self.gate_max
        )
        if is_main_process():
          print(f"  - Gradient gating network initialized (channels: {block_channels})")

      # 5. Proposed visual-to-text semantic anchor.
      # The anchor is built on the 128-d metric feature used by the episodic GNN.
      if self.semantic_anchor_enabled:
        self.semantic_anchor = TextGroundedSemanticAnchor(
          visual_dim=self.metric_input_dim,
          text_dim=text_dim,
          temperature=self.semantic_anchor_temperature,
        ).to(self.device)
        if is_main_process():
          print(
            f"  - Semantic anchor initialized (visual {self.metric_input_dim} -> text {text_dim})"
          )
      
      # 5. [v2.1 鏂规B] 鍙涔犵殑 Text鈫扸isual 鎶曞奖灞?
      # 灏咰LIP text embedding (512-d) 鏄犲皠鍒?visual feature space (128-d)
      if self.text_calibration_weight > 0:
        self.text_to_visual_proj = nn.Sequential(
          nn.Linear(text_dim, 256),
          nn.ReLU(inplace=True),
          nn.Linear(256, self.metric_input_dim),  # 128
        )
        # 鍒濆鍖栦负灏忓€硷紝閬垮厤璁粌鍒濇湡text淇″彿杩囧己
        for m in self.text_to_visual_proj.modules():
          if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            if m.bias is not None:
              nn.init.zeros_(m.bias)
        if is_main_process():
          print(f"  - [v2.1] Learnable text_to_visual_proj initialized (512->{self.metric_input_dim})")
      
      if is_main_process():
        print(f"[TG-MSAT] Text guidance modules initialized successfully!")
      
    except Exception as exc:
      warnings.warn(f"[TG-MSAT] Failed to initialize text guidance: {exc}")
      self.text_guide_epsilon = False
      self.text_guide_gradient = False

  def _get_class_prompts(self, class_ids: List[int]) -> List[str]:
    """Build single-template prompts for a list of class ids."""
    class_names = load_class_names(
      dataset_name=self.dataset_name, 
      data_root=self.data_root
    )
    template = default_template(self.dataset_name)
    
    prompts = []
    for cid in class_ids:
      if class_names and cid < len(class_names):
        name = class_names[cid]
      else:
        name = f"class {cid}"
      prompts.append(template.format(name=name))
    return prompts

  def _get_class_names_for_ids(self, class_ids: List[int]) -> List[str]:
    """Resolve canonicalized class names for the provided class ids."""
    all_class_names = load_class_names(
      dataset_name=self.dataset_name,
      data_root=self.data_root
    )
    names = []
    for cid in class_ids:
      if all_class_names and cid < len(all_class_names):
        names.append(all_class_names[cid])
      else:
        names.append(f"class {cid}")
    return names

  @torch.no_grad()
  def _get_episode_text_prototypes(self, class_ids: List[int]) -> torch.Tensor:
    """Return normalized, detached CLIP prototypes in the episode-local class order."""
    if self.text_prompt_encoder is None:
      raise RuntimeError("semantic text prototypes requested before text encoder initialization")
    class_ids = [int(cid) for cid in class_ids]
    cache_key = (
      self.dataset_name,
      tuple(class_ids),
      bool(self.use_prompt_ensemble),
    )
    cached = self._semantic_text_cache.get(cache_key)
    if cached is not None:
      return cached.to(self.device)

    class_names = self._get_class_names_for_ids(class_ids)
    if self.use_prompt_ensemble:
      templates = get_ensemble_templates(self.dataset_name)
      class_embeddings, _ = self.text_prompt_encoder.encode_class_prompts_ensemble(
        class_names, templates
      )
    else:
      prompts = self._get_class_prompts(class_ids)
      class_embeddings, _ = self.text_prompt_encoder.encode_class_prompts(prompts)

    class_embeddings = F.normalize(class_embeddings.float(), dim=-1).detach()
    self._semantic_text_cache[cache_key] = class_embeddings
    return class_embeddings.to(self.device)

  def _compute_semantic_anchor_loss(
    self,
    visual_features: torch.Tensor,
    global_y_episode: torch.Tensor,
    samples_per_class: int,
    *,
    detach_visual: bool,
    freeze_projector_for_attack: bool = False,
    reduction: str = 'mean',
  ) -> torch.Tensor:
    """Compute the episode-local visual-to-text loss with explicit gradient policy."""
    if not self.semantic_anchor_enabled or self.semantic_anchor is None:
      return visual_features.new_zeros(())
    if global_y_episode.ndim != 2:
      raise ValueError("global_y_episode must have shape [n_way, samples_per_class]")
    class_ids = global_y_episode[:, 0].long().tolist()
    text_prototypes = self._get_episode_text_prototypes(class_ids)
    local_targets = torch.arange(
      self.n_way, device=visual_features.device, dtype=torch.long
    ).repeat_interleave(samples_per_class)
    anchor_features = visual_features.detach() if detach_visual else visual_features
    return self.semantic_anchor(
      anchor_features,
      text_prototypes,
      local_targets,
      freeze_projector_for_attack=freeze_projector_for_attack,
      reduction=reduction,
    )

  def _semantic_drift_diagnostics(
    self,
    attacked_losses: torch.Tensor,
    clean_losses: torch.Tensor,
    samples_per_class: int,
  ) -> dict:
    """Summarize sample, support, and query semantic drift for diagnostics."""
    if attacked_losses.ndim != 1 or clean_losses.ndim != 1:
      raise ValueError('semantic loss diagnostics require per-sample losses')
    if attacked_losses.shape != clean_losses.shape:
      raise ValueError('attacked and clean semantic loss vectors must match')
    if samples_per_class <= 0 or attacked_losses.numel() % samples_per_class != 0:
      raise ValueError('semantic loss vector does not match episode layout')

    sample_drift = attacked_losses.detach() - clean_losses.detach()
    episode_n_way = sample_drift.numel() // samples_per_class
    episode_drift = sample_drift.view(episode_n_way, samples_per_class)
    support_width = min(self.n_support, samples_per_class - 1)
    support_drift = episode_drift[:, :support_width].reshape(-1)
    query_drift = episode_drift[:, support_width:].reshape(-1)
    if support_drift.numel() == 0 or query_drift.numel() == 0:
      raise ValueError('episode must contain both support and query samples')
    return {
      'mean': sample_drift.mean(),
      'support': support_drift.mean(),
      'query': query_drift.mean(),
      'positive_ratio': (sample_drift > 0).to(sample_drift.dtype).mean(),
      'sample_mean': sample_drift.mean(),
      'sample_std': sample_drift.std(unbiased=False),
      'sample_min': sample_drift.min(),
      'sample_max': sample_drift.max(),
      'sample_drift': sample_drift,
    }

  @staticmethod
  def _apply_deterministic_style(input_feature, style_mean, style_std):
    """Apply a projected style statistic without the legacy random skip."""
    if isinstance(style_mean, str) or isinstance(style_std, str):
      return input_feature
    feature_size = input_feature.size()
    current_mean, current_std = calc_mean_std(input_feature)
    normalized = (
      input_feature - current_mean.expand(feature_size)
    ) / current_std.expand(feature_size)
    return (
      normalized * style_std.expand(feature_size)
      + style_mean.expand(feature_size)
    )

  def _semantic_attack_stage_gradients(
    self,
    final_feature: torch.Tensor,
    style_mean: torch.Tensor,
    style_std: torch.Tensor,
    global_y_episode: Optional[torch.Tensor],
    samples_per_class: Optional[int],
  ):
    """Get semantic loss and style gradients with the projector frozen in-loop."""
    if (
      not self.semantic_drift_control_enabled
      or global_y_episode is None
      or samples_per_class is None
    ):
      return None, None, None
    semantic_loss = self._compute_semantic_anchor_loss(
      self.fc(final_feature),
      global_y_episode,
      samples_per_class,
      detach_visual=False,
      freeze_projector_for_attack=True,
    )
    semantic_grad_mean, semantic_grad_std = torch.autograd.grad(
      semantic_loss,
      (style_mean, style_std),
      retain_graph=True,
      allow_unused=False,
    )
    return semantic_loss, semantic_grad_mean.detach(), semantic_grad_std.detach()

  def _resolve_text_guidance_weight(self, prompt_stats: Optional[dict] = None) -> float:
    """Resolve a source-agnostic episode-level text guidance weight."""
    if self.text_weight_mode == 'fixed':
      return float(self.text_weight_fixed)
    if self.text_weight_mode == 'agreement':
      agreement = None if prompt_stats is None else prompt_stats.get("task_agreement")
      if agreement is None:
        return float(1.0 / (1.0 + self.text_weight_k * self.n_support))
      weight = float(agreement) * float(self.text_weight_fixed)
      return max(float(self.text_confidence_floor), min(1.0, weight))
    return float(1.0 / (1.0 + self.text_weight_k * self.n_support))

  @torch.no_grad()
  def _compute_text_guidance(self, global_y: torch.Tensor, samples_per_class: int):
    """
    璁＄畻鏂囨湰寮曞淇℃伅 (涓嶅弬涓庢搴﹁绠?
    
    [Step1: PromptEnsemble] 鏀寔澶氭ā鏉块泦鎴?
      - use_prompt_ensemble=True: 浣跨敤8涓ā鏉跨殑骞冲潎宓屽叆
      - use_prompt_ensemble=False: 浣跨敤鍗曚竴妯℃澘 (鍘熷琛屼负)
    
    [Phase 1 浼樺寲] 娣诲姞 N-shot 鑷€傚簲鏉冮噸:
      - 1-shot: 鏂囨湰鏉冮噸楂?(~0.83)锛屾洿渚濊禆璇箟鍏堥獙
      - 5-shot: 鏂囨湰鏉冮噸浣?(~0.50)锛屾洿鐩镐俊鏁版嵁鏈韩
    
    Returns:
      epsilon_scales: [3] epsilon缂╂斁鍥犲瓙 (濡傛灉text_guide_epsilon寮€鍚?
      gradient_gates: List[Tensor] 姊害闂ㄦ帶 (濡傛灉text_guide_gradient寮€鍚?
    """
    epsilon_scales = None
    gradient_gates = None
    
    if not (self.text_guide_epsilon or self.text_guide_gradient):
      return epsilon_scales, gradient_gates
    
    if self.text_prompt_encoder is None or self.style_prompt_generator is None:
      return epsilon_scales, gradient_gates
    
    try:
      # 1. 鎻愬彇褰撳墠episode鐨勭被鍒獻D
      if global_y.dim() == 1:
        global_y = global_y.view(self.n_way, samples_per_class)
      class_ids = global_y[:, 0].long().tolist()
      
      # 2. 鑾峰彇鏂囨湰宓屽叆 (鏀寔澶氭ā鏉块泦鎴?
      prompt_stats = None
      if self.use_prompt_ensemble:
        # [Step1: PromptEnsemble] Multi-template prompt ensemble.
        class_names = self._get_class_names_for_ids(class_ids)
        templates = get_ensemble_templates(self.dataset_name)
        if not self._warned_missing_prompts:
          if is_main_process():
            print(f"[Step1] Using PromptEnsemble: {len(templates)} templates x {len(class_names)} classes")
          self._warned_missing_prompts = True
        if self.text_weight_mode == 'agreement':
          class_embeddings, z_text, prompt_stats = self.text_prompt_encoder.encode_class_prompts_ensemble_with_stats(
            class_names, templates
          )
        else:
          class_embeddings, z_text = self.text_prompt_encoder.encode_class_prompts_ensemble(
            class_names, templates
          )
      else:
        # Original single-template prompt mode.
        prompts = self._get_class_prompts(class_ids)
        if not prompts:
          if not self._warned_missing_prompts:
            warnings.warn(f"[TG-MSAT] No class prompts found for {self.dataset_name}")
            self._warned_missing_prompts = True
          return epsilon_scales, gradient_gates
        if not self._warned_missing_prompts:
          if is_main_process():
            print(f"[Step1] Using single template (PromptEnsemble OFF)")
          self._warned_missing_prompts = True
        class_embeddings, z_text = self.text_prompt_encoder.encode_class_prompts(prompts)
      
      # 3. 鐢熸垚椋庢牸鎻愮ず
      style_prompt = self.style_prompt_generator(z_text.to(self.device))
      
      # ========== [娑堣瀺瀹為獙] 鍙厤缃殑鏂囨湰寮曞鏉冮噸 ==========
      text_weight = self._resolve_text_guidance_weight(prompt_stats)
      self.last_text_guidance_weight = float(text_weight)
      self.last_text_prompt_agreement = float(
        1.0 if prompt_stats is None else prompt_stats.get("task_agreement", 1.0)
      )
      
      # 5. 璁＄畻epsilon缂╂斁鍥犲瓙 (濡傛灉寮€鍚?
      if self.text_guide_epsilon and self.epsilon_scale_head is not None:
        raw_scales = self.epsilon_scale_head(style_prompt)  # [3], range [0, 1]
        # 鍙厤缃殑epsilon缂╂斁鑼冨洿 [epsilon_scale_min, epsilon_scale_max]
        scale_range = self.epsilon_scale_max - self.epsilon_scale_min
        text_scales = self.epsilon_scale_min + scale_range * raw_scales
        # 搴旂敤 text_weight: 鏈€缁?= text_weight * text_scales + (1 - text_weight) * 1.0
        epsilon_scales = text_weight * text_scales + (1.0 - text_weight) * torch.ones_like(text_scales)
      
      # 6. 璁＄畻姊害闂ㄦ帶 (濡傛灉寮€鍚?
      if self.text_guide_gradient and self.gradient_gate_network is not None:
        gradient_gates = self.gradient_gate_network(style_prompt)  # List of [1, C, 1, 1]
        # 瀵规搴﹂棬鎺т篃搴旂敤鑷€傚簲鏉冮噸
        # gate_final = text_weight * gate + (1 - text_weight) * 1.0
        gradient_gates = [
          text_weight * gate + (1.0 - text_weight) * torch.ones_like(gate)
          for gate in gradient_gates
        ]
      
    except Exception as exc:
      raise RuntimeError(f"[TG-MSAT] Text guidance computation failed: {exc}") from exc
    
    return epsilon_scales, gradient_gates

  def cuda(self):
    """Keep compatibility with calls to .cuda(); set device to CUDA if available, otherwise CPU."""
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
      self.to(self.device)
    except Exception:
      pass
    try:
      self.support_label = self.support_label.to(self.device)
    except Exception:
      pass
    # 绉诲姩鏂囨湰寮曞妯″潡
    for module in [self.text_prompt_encoder, self.style_prompt_generator, 
                   self.epsilon_scale_head, self.gradient_gate_network,
                   self.semantic_anchor]:
      if module is not None:
        try:
          module.to(self.device)
        except Exception:
          pass
    return self

  @torch.no_grad()
  def _get_text_embeddings_for_classes(self, class_ids: List[int]) -> Optional[torch.Tensor]:
    """
    [Step1.1: Test-time Text Calibration]
    鑾峰彇鎸囧畾绫诲埆鐨勬枃鏈祵鍏ワ紝鐢ㄤ簬鎺ㄧ悊鏃剁殑鍒嗘暟鏍″噯銆?
    
    Args:
      class_ids: 绫诲埆ID鍒楄〃
    
    Returns:
      class_embeddings: [n_way, text_dim] 鎴?None
    """
    if self.text_prompt_encoder is None:
      return None
    
    # 浣跨敤缂撳瓨閬垮厤閲嶅璁＄畻
    cache_key = tuple(sorted(class_ids))
    if cache_key in self._text_calibration_cache:
      return self._text_calibration_cache[cache_key]
    
    try:
      class_names = self._get_class_names_for_ids(class_ids)
      
      if self.use_prompt_ensemble:
        templates = get_ensemble_templates(self.dataset_name)
        class_embeddings, _ = self.text_prompt_encoder.encode_class_prompts_ensemble(
          class_names, templates
        )
      else:
        template = default_template(self.dataset_name)
        prompts = [template.format(name=name) for name in class_names]
        class_embeddings, _ = self.text_prompt_encoder.encode_class_prompts(prompts)
      
      # 缂撳瓨缁撴灉
      self._text_calibration_cache[cache_key] = class_embeddings
      return class_embeddings
      
    except Exception as e:
      if not self._warned_missing_prompts:
        warnings.warn(f"[Step1.1] Text calibration failed: {e}")
        self._warned_missing_prompts = True
      return None

  def _apply_text_calibration_v2_embedding(self, z: torch.Tensor, 
                                             class_names: Optional[List[str]] = None) -> torch.Tensor:
    """
    [Step2: Embedding-level Text Calibration - 鏂规1]
    鍦╡mbedding灞傝瀺鍚堟枃鏈俊鎭紝鑰岄潪鍦╨ogits灞傚井璋冦€?
    
    鏍稿績鏀硅繘锛?
    1. 鑾峰彇CLIP text embeddings [n_way, 512]
    2. 鎶曞奖鍒皏isual feature绌洪棿 [n_way, feat_dim=128]
    3. 鐢╲isual-text alignment璋冨埗support embeddings
    4. 璁〨NN鍦?鏂囨湰澧炲己"鐨勭壒寰佷笂鍋氬喅绛?
    
    鍏抽敭鍖哄埆vs v1锛坙ogits灞傦級:
    - v1: text鍙兘寰皟GNN杈撳嚭锛堝奖鍝嶅お寮憋級
    - v2: text鐩存帴鏀瑰彉feature琛ㄧず锛堝奖鍝岹NN鏁翠釜鎺ㄧ悊杩囩▼锛?
    
    Args:
      z: visual features [n_way, n_support+n_query, feat_dim=128]
      class_names: 绫诲埆鍚嶇О鍒楄〃
    
    Returns:
      z_calibrated: 鏂囨湰澧炲己鍚庣殑features [n_way, n_support+n_query, feat_dim]
    """
    if self.text_calibration_weight <= 0 or self.text_prompt_encoder is None:
      return z
    
    if class_names is None or len(class_names) == 0:
      return z
    
    # [Debug] 棣栨璋冪敤鏃舵墦鍗扮‘璁や俊鎭?
    if not hasattr(self, '_calibration_v2_logged'):
      if is_main_process():
        print(f"[Step2 v2] Embedding-level Calibration ACTIVATED: weight={self.text_calibration_weight}, classes={class_names[:3]}...")
      self._calibration_v2_logged = True
    
    try:
      # 1. 鑾峰彇CLIP text embeddings [n_way, 512]
      text_emb = self._get_text_embeddings_for_names(class_names)
      if text_emb is None:
        return z
      text_emb = text_emb.to(self.device)  # [n_way, 512]
      text_emb_norm = F.normalize(text_emb, dim=-1)  # 褰掍竴鍖?
      
      # 2. 鎶曞奖text鍒皏isual feature绌洪棿
      # [v2.1] 浣跨敤鍙涔犵殑鎶曞奖灞傦紙鍦ㄨ缁冩椂浼氭帴鏀舵搴︼級
      if not hasattr(self, 'text_to_visual_proj') or self.text_to_visual_proj is None:
        # 鍥為€€锛氬鏋滄病鏈夊垵濮嬪寲锛屽垱寤轰复鏃跺浐瀹氱殑
        feat_dim = z.size(-1)
        self.text_to_visual_proj = nn.Linear(512, feat_dim, bias=False).to(self.device)
        nn.init.xavier_uniform_(self.text_to_visual_proj.weight)
      
      text_feats = self.text_to_visual_proj(text_emb_norm)  # [n_way, 128]
      text_feats_norm = F.normalize(text_feats, dim=-1)
      
      # 3. 璁＄畻support prototype锛堟瘡涓被鐨剆upport鏍锋湰鍧囧€硷級
      support_z = z[:, :self.n_support, :]  # [n_way, n_support, 128]
      support_proto = support_z.mean(dim=1)  # [n_way, 128]
      support_proto_norm = F.normalize(support_proto, dim=-1)
      
      # 4. 璁＄畻visual-text alignment锛堟瘡涓被鐨剆upport涓庡叾text鐨勭浉浼煎害锛?
      # [n_way, 128] 路 [n_way, 128] -> [n_way]
      visual_text_sim = (support_proto_norm * text_feats_norm).sum(dim=-1)  # cosine similarity
      
      # 5. 鐢╝lignment璋冨埗support features
      # 瀵逛簬alignment楂樼殑绫诲埆锛屽寮哄叾text component
      # text_boost: [n_way] -> [n_way, 1, 1]
      text_boost = torch.sigmoid(visual_text_sim * 2.0)  # scale鍒?0,1), temperature=2
      text_boost = text_boost.unsqueeze(1).unsqueeze(2)  # [n_way, 1, 1]
      
      # 6. 铻嶅悎: z_new = z + alpha * boost * text_feat
      w = self.text_calibration_weight
      text_component = text_feats_norm.unsqueeze(1) * text_boost  # [n_way, 1, 128]
      
      # 鍙皟鍒秙upport features锛坬uery淇濇寔涓嶅彉锛?
      support_calibrated = support_z + w * text_component
      
      # 閲嶆柊缁勫悎
      z_calibrated = torch.cat([
        support_calibrated,
        z[:, self.n_support:, :]  # query涓嶅彉
      ], dim=1)
      
      return z_calibrated
      
    except Exception as e:
      if not self._warned_missing_prompts:
        warnings.warn(f"[Step2 v2] Embedding calibration error: {e}")
        self._warned_missing_prompts = True
      return z

  def _apply_text_calibration_v2_train(self, z: torch.Tensor, 
                                        class_names: Optional[List[str]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    [v2.1 鏂规B] 璁粌鏃剁殑Text Calibration - 甯﹀榻愭崯澶?
    
    涓庢祴璇曟椂鐨勫尯鍒細
    1. 涓嶄娇鐢ˊtorch.no_grad()锛岃projection灞傛帴鏀舵搴?
    2. 杩斿洖棰濆鐨則ext-visual瀵归綈鎹熷け锛岀敤浜庤缁僷rojection灞?
    
    Args:
      z: visual features [n_way, n_support+n_query, feat_dim=128]
      class_names: 绫诲埆鍚嶇О鍒楄〃
    
    Returns:
      z_calibrated: 鏂囨湰澧炲己鍚庣殑features
      loss_align: text-visual瀵归綈鎹熷け
    """
    device = z.device
    loss_align = torch.tensor(0.0, device=device)
    
    if self.text_calibration_weight <= 0 or self.text_prompt_encoder is None:
      return z, loss_align
    
    if class_names is None or len(class_names) == 0:
      return z, loss_align
    
    try:
      # 1. 鑾峰彇CLIP text embeddings [n_way, 512] (杩欓儴鍒嗕笉闇€瑕佹搴?
      with torch.no_grad():
        text_emb = self._get_text_embeddings_for_names(class_names)
        if text_emb is None:
          return z, loss_align
        text_emb = text_emb.to(device)  # [n_way, 512]
        text_emb_norm = F.normalize(text_emb, dim=-1)
      
      # 2. 鎶曞奖text鍒皏isual feature绌洪棿 (杩欓噷闇€瑕佹搴︼紒)
      text_feats = self.text_to_visual_proj(text_emb_norm)  # [n_way, 128]
      text_feats_norm = F.normalize(text_feats, dim=-1)
      
      # 3. 璁＄畻support prototype
      support_z = z[:, :self.n_support, :]  # [n_way, n_support, 128]
      support_proto = support_z.mean(dim=1)  # [n_way, 128]
      support_proto_norm = F.normalize(support_proto, dim=-1)
      
      # 4. 璁＄畻瀵归綈鎹熷け锛氳鎶曞奖鍚庣殑text features涓巚isual prototypes瀵归綈
      # 浣跨敤cosine similarity鎹熷け: 鍚岀被搴旇鐩镐技 (maximize), 涓嶅悓绫诲簲璇ヤ笉鐩镐技 (minimize)
      # [n_way, 128] @ [n_way, 128].T -> [n_way, n_way]
      sim_matrix = support_proto_norm @ text_feats_norm.T  # [n_way, n_way]
      
      # 瀵硅绾挎槸鍚岀被鐩镐技搴︼紙搴旇楂橈級锛岄潪瀵硅绾挎槸寮傜被鐩镐技搴︼紙搴旇浣庯級
      # 浣跨敤InfoNCE椋庢牸鐨勫姣旀崯澶?
      temperature = 0.1
      sim_matrix = sim_matrix / temperature
      labels = torch.arange(self.n_way, device=device)
      loss_align = F.cross_entropy(sim_matrix, labels)
      
      # 5. 璁＄畻visual-text alignment
      visual_text_sim = (support_proto_norm * text_feats_norm).sum(dim=-1)  # [n_way]
      
      # 6. 鐢╝lignment璋冨埗support features
      text_boost = torch.sigmoid(visual_text_sim * 2.0)
      text_boost = text_boost.unsqueeze(1).unsqueeze(2)  # [n_way, 1, 1]
      
      # 7. 铻嶅悎
      w = self.text_calibration_weight
      text_component = text_feats_norm.unsqueeze(1) * text_boost  # [n_way, 1, 128]
      support_calibrated = support_z + w * text_component
      
      # 閲嶆柊缁勫悎
      z_calibrated = torch.cat([
        support_calibrated,
        z[:, self.n_support:, :]  # query涓嶅彉
      ], dim=1)
      
      return z_calibrated, loss_align
      
    except Exception as e:
      if not self._warned_missing_prompts:
        warnings.warn(f"[v2.1] Training calibration error: {e}")
        self._warned_missing_prompts = True
      return z, loss_align
  
  @torch.no_grad()
  def _get_text_embeddings_for_names(self, class_names: List[str]) -> Optional[torch.Tensor]:
    """
    [Step1.1] 鐩存帴閫氳繃绫诲悕鑾峰彇鏂囨湰宓屽叆
    """
    if self.text_prompt_encoder is None:
      return None
    
    # 浣跨敤缂撳瓨
    cache_key = tuple(class_names)
    if cache_key in self._text_calibration_cache:
      return self._text_calibration_cache[cache_key]
    
    try:
      if self.use_prompt_ensemble:
        templates = get_ensemble_templates(self.dataset_name)
        class_embeddings, _ = self.text_prompt_encoder.encode_class_prompts_ensemble(
          class_names, templates
        )
      else:
        template = default_template(self.dataset_name)
        prompts = [template.format(name=name) for name in class_names]
        class_embeddings, _ = self.text_prompt_encoder.encode_class_prompts(prompts)
      
      self._text_calibration_cache[cache_key] = class_embeddings
      return class_embeddings
      
    except Exception as e:
      if not self._warned_missing_prompts:
        warnings.warn(f"[Step1.1] Get text embeddings failed: {e}")
        self._warned_missing_prompts = True
      return None

  def set_forward(self, x, is_feature=False, global_y=None, class_names=None, use_gnn=True):
    self.last_eval_episode_stats = {}
    x = x.to(self.device, non_blocking=True)

    if is_feature:
      expected_samples = self.n_support + self.n_query
      assert x.size(1) == expected_samples
      raw_episode_features = x
      z = self.fc(x.view(-1, *x.size()[2:]))
      z = z.view(self.n_way, -1, z.size(1))
    else:
      # get feature using encoder
      x = x.view(-1, *x.size()[2:])
      raw_flat = self.feature(x)
      raw_episode_features = raw_flat.view(self.n_way, -1, raw_flat.size(-1))
      z = self.fc(raw_flat)
      z = z.view(self.n_way, -1, z.size(1))

    # [Step2: Embedding-level Text Calibration]
    # 鍦℅NN涔嬪墠瀵筫mbedding杩涜鏂囨湰澧炲己锛堣€岄潪鍦╨ogits灞傚井璋冿級
    if self.text_calibration_weight > 0 and class_names is not None:
      z = self._apply_text_calibration_v2_embedding(z, class_names)

    eval_stats = {}
    refined_prototypes = None
    if self.eval_episode_relevance or self.eval_proto_refine or not use_gnn:
      z_support = z[:, :self.n_support]
      z_query = z[:, self.n_support:]
      base_prototypes = z_support.mean(dim=1)
      query_metric_features = z_query.reshape(-1, z.size(-1))
      raw_query_features = raw_episode_features[:, self.n_support:].reshape(-1, raw_episode_features.size(-1))
      global_selected_mask = self._build_eval_global_selection_mask(raw_query_features)
      refined_prototypes, _, eval_stats = self._refine_episode_prototypes(
        base_prototypes,
        query_metric_features,
        global_selected_mask,
        support_metric_features=z_support,
        allow_refine=(not use_gnn),
      )
      self.last_eval_episode_stats = eval_stats

    if use_gnn:
      # GNN璺緞: 浣跨敤鍥剧缁忕綉缁滆繘琛屾爣绛句紶鎾垎绫?
      # stack the feature for metric function: n_way * n_s + n_q * f -> n_q * [1 * n_way(n_s + 1) * f]
      z_stack = self._build_query_episode_stack(z)
      assert z_stack.size(1) == self.n_way * (self.n_support + 1)
      scores = self.forward_gnn(z_stack).float()
    else:
      # ProtoNet璺緞: 浣跨敤浣欏鸡鐩镐技搴︽渶杩戝師鍨嬪垎绫?(鐢ㄤ簬GNN娑堣瀺)
      z_support = z[:, :self.n_support]           # [n_way, n_s, d]
      z_query = z[:, self.n_support:]              # [n_way, n_q, d]
      prototypes = refined_prototypes if refined_prototypes is not None else z_support.mean(dim=1)
      z_q_flat = z_query.reshape(-1, z.size(-1))   # [n_way*n_q, d]
      scores = self._compute_proto_scores(z_q_flat, prototypes)
    
    return scores


  def _compute_proto_scores(self, query_features, prototypes, logit_scale=10.0):
    query_norm = F.normalize(query_features, dim=-1)
    proto_norm = F.normalize(prototypes, dim=-1)
    return (query_norm @ proto_norm.t()) * float(logit_scale)

  def _resolve_eval_proto_refine_alpha(self):
    if self.n_support <= 1:
      return float(self.eval_proto_refine_alpha_1shot)
    return float(self.eval_proto_refine_alpha_5shot)

  def _resolve_eval_proto_refine_thresholds(self):
    if self.n_support <= 1:
      return {
        "affinity_threshold": float(self.eval_proto_refine_affinity_threshold_1shot),
        "margin_threshold": float(self.eval_proto_refine_margin_threshold_1shot),
        "min_selected": int(self.eval_proto_refine_min_selected_1shot),
        "min_coverage": float(self.eval_proto_refine_min_coverage_1shot),
        "topk": int(max(self.eval_proto_refine_topk_1shot, 0)),
        "target_affinity_gain": float(self.eval_proto_refine_target_affinity_gain_1shot),
        "target_margin_gain": float(self.eval_proto_refine_target_margin_gain_1shot),
      }
    return {
      "affinity_threshold": float(self.eval_proto_refine_affinity_threshold_5shot),
      "margin_threshold": float(self.eval_proto_refine_margin_threshold_5shot),
      "min_selected": int(self.eval_proto_refine_min_selected_5shot),
      "min_coverage": float(self.eval_proto_refine_min_coverage_5shot),
      "topk": 0,
      "target_affinity_gain": float(self.eval_proto_refine_target_affinity_gain_5shot),
      "target_margin_gain": float(self.eval_proto_refine_target_margin_gain_5shot),
    }

  def _build_eval_global_selection_mask(self, query_global_features):
    if query_global_features is None or query_global_features.numel() == 0:
      empty = torch.zeros(0, dtype=torch.bool, device=self.device)
      return empty

    logits = self.classifier.forward(query_global_features)
    probs = torch.softmax(logits / max(self.target_ssl_temperature, 1e-6), dim=1)
    conf = probs.max(dim=1).values
    log_class_count = math.log(max(probs.size(1), 2))
    entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1) / log_class_count
    return (conf >= self.target_ssl_confidence_threshold) & (entropy <= self.target_ssl_max_entropy)

  def _summarize_episode_subset(self, mask, affinity, margin, pseudo_labels):
    if mask.numel() == 0:
      return {
        "count": 0.0,
        "ratio": 0.0,
        "prototype_affinity": 0.0,
        "prototype_margin": 0.0,
        "class_coverage": 0.0,
      }

    mask = mask.bool()
    count = int(mask.sum().item())
    ratio = float(mask.float().mean().item())
    if count <= 0:
      return {
        "count": 0.0,
        "ratio": ratio,
        "prototype_affinity": 0.0,
        "prototype_margin": 0.0,
        "class_coverage": 0.0,
      }

    coverage = float(pseudo_labels[mask].unique().numel()) / float(max(self.n_way, 1))
    return {
      "count": float(count),
      "ratio": ratio,
      "prototype_affinity": float(affinity[mask].mean().item()),
      "prototype_margin": float(margin[mask].mean().item()),
      "class_coverage": coverage,
    }

  def _compute_proto_margin(self, scores):
    if scores.size(1) <= 1:
      return 1.0
    top2 = scores.topk(k=2, dim=1).values
    return float((top2[:, 0] - top2[:, 1]).mean().item())

  def _summarize_query_subset_with_prototypes(self, mask, query_metric_features, prototypes):
    if query_metric_features.numel() == 0:
      return {
        "count": 0.0,
        "ratio": 0.0,
        "prototype_affinity": 0.0,
        "prototype_margin": 0.0,
        "class_coverage": 0.0,
      }

    scores = self._compute_proto_scores(query_metric_features, prototypes)
    probs = torch.softmax(scores, dim=1)
    topk = scores.topk(k=min(2, scores.size(1)), dim=1)
    affinity = topk.values[:, 0]
    if scores.size(1) > 1:
      margin = topk.values[:, 0] - topk.values[:, 1]
    else:
      margin = torch.ones_like(affinity)
    pseudo_labels = probs.argmax(dim=1)
    return self._summarize_episode_subset(mask, affinity, margin, pseudo_labels)

  def _propose_refined_prototypes(self, base_prototypes, query_metric_features, global_selected_mask):
    thresholds = self._resolve_eval_proto_refine_thresholds()
    stats = {
      "all_selected_count": 0.0,
      "all_selected_ratio": 0.0,
      "all_selected_prototype_affinity": 0.0,
      "all_selected_prototype_margin": 0.0,
      "all_selected_class_coverage": 0.0,
      "gate_passed_count": 0.0,
      "gate_passed_ratio": 0.0,
      "gate_passed_prototype_affinity": 0.0,
      "gate_passed_prototype_margin": 0.0,
      "gate_passed_class_coverage": 0.0,
      "target_prototype_affinity_before": 0.0,
      "target_prototype_affinity_candidate": 0.0,
      "target_prototype_margin_before": 0.0,
      "target_prototype_margin_candidate": 0.0,
      "target_class_coverage_before": 0.0,
      "target_class_coverage_candidate": 0.0,
      "refinement_candidate": 0.0,
      "refinement_anchor": self._resolve_eval_proto_refine_alpha(),
      "refinement_topk_cap": float(thresholds["topk"]),
    }
    if query_metric_features.numel() == 0:
      return base_prototypes, stats

    proto_scores = self._compute_proto_scores(query_metric_features, base_prototypes)
    proto_logits = proto_scores / 10.0
    proto_probs = torch.softmax(proto_scores, dim=1)
    topk = proto_logits.topk(k=min(2, proto_logits.size(1)), dim=1)
    affinity = topk.values[:, 0]
    if proto_logits.size(1) > 1:
      margin = topk.values[:, 0] - topk.values[:, 1]
    else:
      margin = torch.ones_like(affinity)
    pseudo_labels = proto_probs.argmax(dim=1)

    all_mask = global_selected_mask.bool()
    gate_mask = all_mask & (affinity >= thresholds["affinity_threshold"]) & (margin >= thresholds["margin_threshold"])

    if thresholds["topk"] > 0 and int(gate_mask.sum().item()) > thresholds["topk"]:
      gated_indices = torch.nonzero(gate_mask, as_tuple=False).squeeze(1)
      ranked = affinity[gated_indices]
      topk_count = min(thresholds["topk"], gated_indices.numel())
      keep_local = ranked.topk(k=topk_count, largest=True).indices
      keep_indices = gated_indices[keep_local]
      limited_gate_mask = torch.zeros_like(gate_mask)
      limited_gate_mask[keep_indices] = True
      gate_mask = limited_gate_mask

    all_stats = self._summarize_episode_subset(all_mask, affinity, margin, pseudo_labels)
    gate_stats = self._summarize_episode_subset(gate_mask, affinity, margin, pseudo_labels)
    gate_candidate = (
      gate_stats["count"] >= float(thresholds["min_selected"])
      and gate_stats["class_coverage"] >= thresholds["min_coverage"]
    )

    refined_prototypes = base_prototypes
    before_target_stats = self._summarize_query_subset_with_prototypes(gate_mask, query_metric_features, base_prototypes)
    candidate_target_stats = before_target_stats
    if gate_candidate:
      alpha = self._resolve_eval_proto_refine_alpha()
      gated_probs = proto_probs[gate_mask]
      gated_queries = query_metric_features[gate_mask]
      weighted_sum = gated_probs.t() @ gated_queries
      class_mass = gated_probs.sum(dim=0).unsqueeze(1)
      refined_prototypes = (alpha * base_prototypes + weighted_sum) / (alpha + class_mass).clamp_min(1e-6)
      candidate_target_stats = self._summarize_query_subset_with_prototypes(gate_mask, query_metric_features, refined_prototypes)

    stats.update({
      "all_selected_count": all_stats["count"],
      "all_selected_ratio": all_stats["ratio"],
      "all_selected_prototype_affinity": all_stats["prototype_affinity"],
      "all_selected_prototype_margin": all_stats["prototype_margin"],
      "all_selected_class_coverage": all_stats["class_coverage"],
      "gate_passed_count": gate_stats["count"],
      "gate_passed_ratio": gate_stats["ratio"],
      "gate_passed_prototype_affinity": gate_stats["prototype_affinity"],
      "gate_passed_prototype_margin": gate_stats["prototype_margin"],
      "gate_passed_class_coverage": gate_stats["class_coverage"],
      "target_prototype_affinity_before": before_target_stats["prototype_affinity"],
      "target_prototype_affinity_candidate": candidate_target_stats["prototype_affinity"],
      "target_prototype_margin_before": before_target_stats["prototype_margin"],
      "target_prototype_margin_candidate": candidate_target_stats["prototype_margin"],
      "target_class_coverage_before": before_target_stats["class_coverage"],
      "target_class_coverage_candidate": candidate_target_stats["class_coverage"],
      "refinement_candidate": float(gate_candidate),
    })
    return refined_prototypes, stats

  def _compute_support_leave_one_out_metrics(self, support_metric_features, query_metric_features, global_selected_mask, allow_candidate_refine=False):
    metrics = {
      "support_loo_before": float("nan"),
      "support_loo_candidate": float("nan"),
      "support_margin_before": float("nan"),
      "support_margin_candidate": float("nan"),
    }
    if support_metric_features is None or support_metric_features.numel() == 0:
      return metrics

    if self.n_support <= 1:
      support_points = support_metric_features[:, 0, :]
      base_prototypes = support_metric_features.mean(dim=1)
      before_scores = self._compute_proto_scores(support_points, base_prototypes)
      candidate_prototypes = base_prototypes
      if allow_candidate_refine:
        candidate_prototypes, _ = self._propose_refined_prototypes(
          base_prototypes,
          query_metric_features,
          global_selected_mask,
        )
      candidate_scores = self._compute_proto_scores(support_points, candidate_prototypes)
      support_targets = torch.arange(self.n_way, device=support_points.device)
      metrics["support_loo_before"] = float((before_scores.argmax(dim=1) == support_targets).float().mean().item())
      metrics["support_loo_candidate"] = float((candidate_scores.argmax(dim=1) == support_targets).float().mean().item())
      metrics["support_margin_before"] = self._compute_proto_margin(before_scores)
      metrics["support_margin_candidate"] = self._compute_proto_margin(candidate_scores)
      return metrics

    correct_before = 0
    correct_candidate = 0
    before_margin_sum = 0.0
    candidate_margin_sum = 0.0
    total = 0

    for class_idx in range(self.n_way):
      for support_idx in range(self.n_support):
        held_out = support_metric_features[class_idx, support_idx].unsqueeze(0)
        proto_list = []
        valid = True
        for proto_class in range(self.n_way):
          if proto_class == class_idx:
            remaining = torch.cat(
              [support_metric_features[proto_class, :support_idx], support_metric_features[proto_class, support_idx + 1 :]],
              dim=0,
            )
          else:
            remaining = support_metric_features[proto_class]
          if remaining.size(0) == 0:
            valid = False
            break
          proto_list.append(remaining.mean(dim=0))
        if not valid:
          continue

        base_prototypes = torch.stack(proto_list, dim=0)
        before_scores = self._compute_proto_scores(held_out, base_prototypes)
        before_margin_sum += self._compute_proto_margin(before_scores)
        correct_before += int(before_scores.argmax(dim=1).item() == class_idx)

        candidate_prototypes = base_prototypes
        if allow_candidate_refine:
          candidate_prototypes, _ = self._propose_refined_prototypes(
            base_prototypes,
            query_metric_features,
            global_selected_mask,
          )
        candidate_scores = self._compute_proto_scores(held_out, candidate_prototypes)
        candidate_margin_sum += self._compute_proto_margin(candidate_scores)
        correct_candidate += int(candidate_scores.argmax(dim=1).item() == class_idx)
        total += 1

    if total == 0:
      return metrics

    metrics["support_loo_before"] = float(correct_before) / float(total)
    metrics["support_loo_candidate"] = float(correct_candidate) / float(total)
    metrics["support_margin_before"] = before_margin_sum / float(total)
    metrics["support_margin_candidate"] = candidate_margin_sum / float(total)
    return metrics

  def _refine_episode_prototypes(self, base_prototypes, query_metric_features, global_selected_mask, support_metric_features=None, allow_refine=True):
    if query_metric_features.numel() == 0:
      stats = {
        "all_selected_count": 0.0,
        "all_selected_ratio": 0.0,
        "all_selected_prototype_affinity": 0.0,
        "all_selected_prototype_margin": 0.0,
        "all_selected_class_coverage": 0.0,
        "gate_passed_count": 0.0,
        "gate_passed_ratio": 0.0,
        "gate_passed_prototype_affinity": 0.0,
        "gate_passed_prototype_margin": 0.0,
        "gate_passed_class_coverage": 0.0,
        "refinement_applied": 0.0,
        "refinement_candidate": 0.0,
        "refinement_guard_passed": 1.0,
        "refinement_target_guard_passed": 1.0,
        "refinement_loo_guard_passed": 1.0,
        "refinement_margin_guard_passed": 1.0,
        "refinement_rejected_by_support_guard": 0.0,
        "refinement_rejected_by_target_guard": 0.0,
        "refinement_rejected_by_support_loo_guard": 0.0,
        "refinement_rejected_by_support_margin_guard": 0.0,
        "refinement_anchor": self._resolve_eval_proto_refine_alpha(),
        "refinement_topk_cap": float(self._resolve_eval_proto_refine_thresholds()["topk"]),
        "support_loo_before": float("nan"),
        "support_loo_candidate": float("nan"),
        "support_loo_after": float("nan"),
        "support_margin_before": float("nan"),
        "support_margin_candidate": float("nan"),
        "support_margin_after": float("nan"),
        "target_prototype_affinity_before": 0.0,
        "target_prototype_affinity_candidate": 0.0,
        "target_prototype_affinity_after": 0.0,
        "target_prototype_margin_before": 0.0,
        "target_prototype_margin_candidate": 0.0,
        "target_prototype_margin_after": 0.0,
        "target_class_coverage_before": 0.0,
        "target_class_coverage_candidate": 0.0,
        "target_class_coverage_after": 0.0,
      }
      return base_prototypes, None, stats

    refined_candidate, proposal_stats = self._propose_refined_prototypes(
      base_prototypes,
      query_metric_features,
      global_selected_mask,
    )
    thresholds = self._resolve_eval_proto_refine_thresholds()
    candidate_requested = bool(allow_refine and self.eval_proto_refine and proposal_stats["refinement_candidate"] > 0.5)
    guard_passed = True
    target_guard_passed = True
    loo_guard_passed = True
    margin_guard_passed = True
    rejected_by_support_guard = False
    rejected_by_target_guard = False
    rejected_by_support_loo_guard = False
    rejected_by_support_margin_guard = False
    support_metrics = self._compute_support_leave_one_out_metrics(
      support_metric_features,
      query_metric_features,
      global_selected_mask,
      allow_candidate_refine=candidate_requested,
    ) if support_metric_features is not None else {
      "support_loo_before": float("nan"),
      "support_loo_candidate": float("nan"),
      "support_margin_before": float("nan"),
      "support_margin_candidate": float("nan"),
    }

    if candidate_requested and self.eval_proto_refine_target_guard:
      before_target_affinity = proposal_stats.get("target_prototype_affinity_before", float("nan"))
      candidate_target_affinity = proposal_stats.get("target_prototype_affinity_candidate", float("nan"))
      before_target_margin = proposal_stats.get("target_prototype_margin_before", float("nan"))
      candidate_target_margin = proposal_stats.get("target_prototype_margin_candidate", float("nan"))
      required_target_affinity = before_target_affinity + thresholds["target_affinity_gain"] - self.eval_proto_refine_target_affinity_tolerance
      required_target_margin = before_target_margin + thresholds["target_margin_gain"] - self.eval_proto_refine_target_margin_tolerance
      affinity_ok = (not math.isnan(before_target_affinity)) and (not math.isnan(candidate_target_affinity)) and (
        candidate_target_affinity >= required_target_affinity
      )
      margin_ok = (not math.isnan(before_target_margin)) and (not math.isnan(candidate_target_margin)) and (
        candidate_target_margin >= required_target_margin
      )
      target_guard_passed = affinity_ok and margin_ok
      rejected_by_target_guard = not target_guard_passed

    if candidate_requested and self.eval_proto_refine_support_loo_guard:
      before_loo = support_metrics["support_loo_before"]
      candidate_loo = support_metrics["support_loo_candidate"]
      if (not math.isnan(before_loo)) and (not math.isnan(candidate_loo)):
        loo_guard_passed = candidate_loo >= (before_loo - self.eval_proto_refine_support_loo_tolerance)
      else:
        loo_guard_passed = False
      rejected_by_support_loo_guard = not loo_guard_passed

    if candidate_requested and self.eval_proto_refine_support_margin_guard:
      before_margin = support_metrics["support_margin_before"]
      candidate_margin = support_metrics["support_margin_candidate"]
      if (not math.isnan(before_margin)) and (not math.isnan(candidate_margin)):
        margin_guard_passed = candidate_margin >= (before_margin - self.eval_proto_refine_support_margin_tolerance)
      else:
        margin_guard_passed = False
      rejected_by_support_margin_guard = not margin_guard_passed

    if candidate_requested:
      if self.eval_proto_refine_target_guard:
        guard_passed = guard_passed and target_guard_passed
      if self.eval_proto_refine_support_loo_guard:
        guard_passed = guard_passed and loo_guard_passed
      if self.eval_proto_refine_support_margin_guard:
        guard_passed = guard_passed and margin_guard_passed
      rejected_by_support_guard = not guard_passed

    refine_applied = bool(candidate_requested and guard_passed)
    refined_prototypes = refined_candidate if refine_applied else base_prototypes

    stats = {
      "all_selected_count": proposal_stats["all_selected_count"],
      "all_selected_ratio": proposal_stats["all_selected_ratio"],
      "all_selected_prototype_affinity": proposal_stats["all_selected_prototype_affinity"],
      "all_selected_prototype_margin": proposal_stats["all_selected_prototype_margin"],
      "all_selected_class_coverage": proposal_stats["all_selected_class_coverage"],
      "gate_passed_count": proposal_stats["gate_passed_count"],
      "gate_passed_ratio": proposal_stats["gate_passed_ratio"],
      "gate_passed_prototype_affinity": proposal_stats["gate_passed_prototype_affinity"],
      "gate_passed_prototype_margin": proposal_stats["gate_passed_prototype_margin"],
      "gate_passed_class_coverage": proposal_stats["gate_passed_class_coverage"],
      "refinement_applied": float(refine_applied),
      "refinement_candidate": float(candidate_requested),
      "refinement_guard_passed": float(guard_passed),
      "refinement_target_guard_passed": float(target_guard_passed),
      "refinement_loo_guard_passed": float(loo_guard_passed),
      "refinement_margin_guard_passed": float(margin_guard_passed),
      "refinement_rejected_by_support_guard": float(rejected_by_support_guard),
      "refinement_rejected_by_target_guard": float(rejected_by_target_guard),
      "refinement_rejected_by_support_loo_guard": float(rejected_by_support_loo_guard),
      "refinement_rejected_by_support_margin_guard": float(rejected_by_support_margin_guard),
      "refinement_anchor": self._resolve_eval_proto_refine_alpha(),
      "refinement_topk_cap": proposal_stats.get("refinement_topk_cap", 0.0),
      "support_loo_before": support_metrics["support_loo_before"],
      "support_loo_candidate": support_metrics["support_loo_candidate"],
      "support_loo_after": support_metrics["support_loo_candidate"] if refine_applied else support_metrics["support_loo_before"],
      "support_margin_before": support_metrics["support_margin_before"],
      "support_margin_candidate": support_metrics["support_margin_candidate"],
      "support_margin_after": support_metrics["support_margin_candidate"] if refine_applied else support_metrics["support_margin_before"],
      "target_prototype_affinity_before": proposal_stats.get("target_prototype_affinity_before", 0.0),
      "target_prototype_affinity_candidate": proposal_stats.get("target_prototype_affinity_candidate", 0.0),
      "target_prototype_affinity_after": proposal_stats.get("target_prototype_affinity_candidate", 0.0) if refine_applied else proposal_stats.get("target_prototype_affinity_before", 0.0),
      "target_prototype_margin_before": proposal_stats.get("target_prototype_margin_before", 0.0),
      "target_prototype_margin_candidate": proposal_stats.get("target_prototype_margin_candidate", 0.0),
      "target_prototype_margin_after": proposal_stats.get("target_prototype_margin_candidate", 0.0) if refine_applied else proposal_stats.get("target_prototype_margin_before", 0.0),
      "target_class_coverage_before": proposal_stats.get("target_class_coverage_before", 0.0),
      "target_class_coverage_candidate": proposal_stats.get("target_class_coverage_candidate", 0.0),
      "target_class_coverage_after": proposal_stats.get("target_class_coverage_candidate", 0.0) if refine_applied else proposal_stats.get("target_class_coverage_before", 0.0),
    }
    return refined_prototypes, None, stats



  def forward_gnn(self, zs):
    # gnn inp: n_q * n_way(n_s + 1) * f
    self._ensure_gnn_for_current_way()
    if torch.is_tensor(zs):
      num_queries = zs.size(0)
      support_label = self._build_support_label_dynamic().expand(num_queries, -1, -1)
      nodes = torch.cat([zs, support_label], dim=2)
    else:
      support_label = self._build_support_label_dynamic()
      nodes = torch.cat([torch.cat([z, support_label], dim=2) for z in zs], dim=0)
    scores = self.gnn(nodes).float()

    # n_q * n_way(n_s + 1) * n_way -> (n_way * n_q) * n_way
    scores = scores.view(self.n_query, self.n_way, self.n_support + 1, self.n_way)[:, :, -1].permute(1, 0, 2).contiguous().view(-1, self.n_way)
    return scores

  def _build_support_label_dynamic(self):
    """Build support label buffer on the fly to match current n_way/n_support."""
    support_label = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).long().unsqueeze(1)
    support_label = torch.zeros(self.n_way * self.n_support, self.n_way).scatter(1, support_label, 1).view(self.n_way, self.n_support, self.n_way)
    support_label = torch.cat([support_label, torch.zeros(self.n_way, 1, self.n_way)], dim=1)
    return support_label.view(1, -1, self.n_way).to(self.device)

  def _ensure_gnn_for_current_way(self):
    """Rebuild GNN if n_way changed to keep input/output dims consistent."""
    expected_input = self.metric_dim + self.n_way
    needs_rebuild = (
      not hasattr(self, "gnn")
      or getattr(self.gnn, "input_features", None) != expected_input
      or getattr(self.gnn, "layer_last", None) is None
      or getattr(self.gnn.layer_last, "num_outputs", None) != self.n_way
    )
    if needs_rebuild:
      self.gnn = GNN_nl(expected_input, 96, self.n_way).to(self.device)

  def _build_query_episode_stack(self, z):
    support = z[:, :self.n_support].unsqueeze(0).expand(self.n_query, -1, -1, -1)
    queries = z[:, self.n_support:].permute(1, 0, 2).unsqueeze(2)
    stacked = torch.cat([support, queries], dim=2)
    return stacked.reshape(self.n_query, self.n_way * (self.n_support + 1), z.size(-1))


  def set_forward_loss(self, x, global_y=None):
    y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
    y_query = y_query.long().to(self.device, non_blocking=True)
    scores = self.set_forward(x).float()
    loss = self.loss_fn(scores, y_query)
    return scores, loss


  def adversarial_attack_Incre(
    self,
    x_ori,
    y_ori,
    epsilon_list,
    epsilon_scales=None,
    gradient_gates=None,
    attack_mode='progressive',
    semantic_clean_loss=None,
    semantic_clean_loss_per_sample=None,
    semantic_global_y_episode=None,
    semantic_samples_per_class=None,
  ):
    """
    澧為噺寮忛鏍煎鎶楁敾鍑?(鏀寔鏂囨湰寮曞)
    
    Args:
      x_ori: 鍘熷鍥惧儚
      y_ori: 鏍囩
      epsilon_list: 鍩虹epsilon鍒楄〃 [0.8, 0.08, 0.008]
      epsilon_scales: [鍙€塢 鏂囨湰寮曞鐨別psilon缂╂斁鍥犲瓙 [3]
      gradient_gates: [鍙€塢 鏂囨湰寮曞鐨勬搴﹂棬鎺?List[[1,C,1,1]]
    """
    device = next(self.parameters()).device
    x_ori = x_ori.to(device)
    y_ori = y_ori.to(device)
    x_size = x_ori.size()
    x_ori = x_ori.view(x_size[0]*x_size[1], x_size[2], x_size[3], x_size[4])
    y_ori = y_ori.view(x_size[0]*x_size[1])

    semantic_lambda = x_ori.new_zeros(()).detach()
    self.last_semantic_attack_stats = {}
    if self.semantic_drift_control_enabled:
      missing = []
      if semantic_clean_loss is None:
        missing.append('semantic_clean_loss')
      if semantic_clean_loss_per_sample is None:
        missing.append('semantic_clean_loss_per_sample')
      if semantic_global_y_episode is None:
        missing.append('semantic_global_y_episode')
      if semantic_samples_per_class is None:
        missing.append('semantic_samples_per_class')
      if missing:
        raise ValueError(
          'semantic drift control requires: ' + ', '.join(missing)
        )

    def semantic_style_candidate(base_style, task_gradient, semantic_gradient, epsilon, is_sigma):
      weights = fixed_budget_channel_weights(
        task_gradient,
        semantic_gradient,
        beta=self.semantic_drift_config.semantic_risk_beta,
        temperature=self.semantic_drift_config.budget_temperature,
        delta=self.semantic_drift_config.stability_delta,
        channel_dim=1,
        detach=True,
      )
      objective_gradient = task_gradient - semantic_lambda.detach() * semantic_gradient
      candidate = weighted_sign_step(
        base_style,
        objective_gradient,
        weights,
        step_size=float(epsilon),
        detach_direction=True,
      )
      return project_style_budget(
        candidate,
        base_style,
        float(epsilon),
        channel_dim=1,
        minimum=self.semantic_drift_config.sigma_min if is_sigma else None,
        delta=self.semantic_drift_config.stability_delta,
      )

    def record_semantic_stage(
      stage,
      semantic_loss,
      diagnostics,
      lambda_before,
      stats_mean,
      stats_std,
    ):
      if not self.semantic_drift_control_enabled:
        return
      self.last_semantic_attack_stats[f'stage_{stage}'] = {
        'semantic_loss': float(semantic_loss.detach().cpu()),
        'drift': float(diagnostics['mean'].detach().cpu()),
        'support_drift': float(diagnostics['support'].detach().cpu()),
        'query_drift': float(diagnostics['query'].detach().cpu()),
        'positive_drift_ratio': float(diagnostics['positive_ratio'].detach().cpu()),
        'sample_drift_mean': float(diagnostics['sample_mean'].detach().cpu()),
        'sample_drift_std': float(diagnostics['sample_std'].detach().cpu()),
        'sample_drift_min': float(diagnostics['sample_min'].detach().cpu()),
        'sample_drift_max': float(diagnostics['sample_max'].detach().cpu()),
        'sample_drift': [
          float(value) for value in diagnostics['sample_drift'].detach().cpu().tolist()
        ],
        'lambda_before': float(lambda_before.detach().cpu()),
        'lambda_after': float(semantic_lambda.detach().cpu()),
        'mean': {k: float(v.detach().cpu()) for k, v in stats_mean.items()},
        'sigma': {k: float(v.detach().cpu()) for k, v in stats_std.items()},
      }

    # if not adv, set defalut = 'None'
    adv_style_mean_block1, adv_style_std_block1 = 'None', 'None'
    adv_style_mean_block2, adv_style_std_block2 = 'None', 'None'
    adv_style_mean_block3, adv_style_std_block3 = 'None', 'None'

    # ========== 鏂囨湰寮曞: 璋冩暣epsilon ==========
    if epsilon_scales is not None:
      # epsilon_scales: [3], range [0.5, 1.5]
      scaled_epsilon_list = [
        epsilon_list[0] * epsilon_scales[0].item(),
        epsilon_list[1] * epsilon_scales[1].item(),
        epsilon_list[2] * epsilon_scales[2].item(),
      ]
    else:
      scaled_epsilon_list = epsilon_list

    # forward and set the grad = True
    blocklist = 'block123'
    use_progressive = (attack_mode == 'progressive')
    
    if('1' in blocklist and scaled_epsilon_list[0] != 0 ):
      # forward block1
      x_ori_block1_raw = self.feature.forward_block1(x_ori)
      x_ori_block1 = x_ori_block1_raw
      feat_size_block1 = x_ori_block1.size()
      ori_style_mean_block1, ori_style_std_block1 = calc_mean_std(x_ori_block1)
      # set them as learnable parameters
      ori_style_mean_block1  = torch.nn.Parameter(ori_style_mean_block1)
      ori_style_std_block1 = torch.nn.Parameter(ori_style_std_block1)
      ori_style_mean_block1.requires_grad_()
      ori_style_std_block1.requires_grad_()
      # contain ori_style_mean_block1 in the graph 
      x_normalized_block1 = (x_ori_block1 - ori_style_mean_block1.detach().expand(feat_size_block1)) / ori_style_std_block1.detach().expand(feat_size_block1)
      x_ori_block1 = x_normalized_block1 * ori_style_std_block1.expand(feat_size_block1) + ori_style_mean_block1.expand(feat_size_block1)
      
      # pass the rest model
      x_ori_block2 = self.feature.forward_block2(x_ori_block1)
      x_ori_block3 = self.feature.forward_block3(x_ori_block2)
      x_ori_block4 = self.feature.forward_block4(x_ori_block3)
      x_ori_fea = self.feature.forward_rest(x_ori_block4)
      x_ori_output = self.classifier.forward(x_ori_fea)
    
      # calculate initial pred, loss and acc
      ori_pred = x_ori_output.max(1, keepdim=True)[1]
      ori_loss = self.loss_fn(x_ori_output, y_ori)
      ori_acc = (ori_pred == y_ori).type(torch.float).sum().item() / y_ori.size()[0]

      if self.semantic_drift_control_enabled:
        task_grad_mean, task_grad_std = torch.autograd.grad(
          ori_loss,
          (ori_style_mean_block1, ori_style_std_block1),
          retain_graph=True,
        )
        semantic_loss_stage, semantic_grad_mean, semantic_grad_std = self._semantic_attack_stage_gradients(
          x_ori_fea,
          ori_style_mean_block1,
          ori_style_std_block1,
          semantic_global_y_episode,
          semantic_samples_per_class,
        )
        epsilon = float(epsilon_list[0])
        lambda_before = semantic_lambda.detach()
        adv_style_mean_block1, mean_stats = semantic_style_candidate(
          ori_style_mean_block1.detach(), task_grad_mean.detach(), semantic_grad_mean, epsilon, False
        )
        adv_style_std_block1, std_stats = semantic_style_candidate(
          ori_style_std_block1.detach(), task_grad_std.detach(), semantic_grad_std, epsilon, True
        )
        with torch.no_grad():
          post_block1 = self._apply_deterministic_style(
            x_ori_block1_raw.detach(), adv_style_mean_block1, adv_style_std_block1
          )
          post_block2 = self.feature.forward_block2(post_block1)
          post_block3 = self.feature.forward_block3(post_block2)
          post_block4 = self.feature.forward_block4(post_block3)
          post_fea = self.feature.forward_rest(post_block4)
          post_semantic_losses = self._compute_semantic_anchor_loss(
            self.fc(post_fea),
            semantic_global_y_episode,
            semantic_samples_per_class,
            detach_visual=True,
            freeze_projector_for_attack=True,
            reduction='none',
          )
          post_semantic_loss = post_semantic_losses.mean()
          diagnostics = self._semantic_drift_diagnostics(
            post_semantic_losses,
            semantic_clean_loss_per_sample,
            semantic_samples_per_class,
          )
          stage_drift = diagnostics['mean']
        semantic_lambda = update_dual_state(
          semantic_lambda, stage_drift, self.semantic_drift_config
        )
        record_semantic_stage(
          1, post_semantic_loss, diagnostics, lambda_before, mean_stats, std_stats
        )
      else:
        # Preserve the original StyleAdv random-start FGSM branch exactly.
        self.feature.zero_grad()
        self.classifier.zero_grad()
        ori_loss.backward()
        grad_ori_style_mean_block1 = ori_style_mean_block1.grad.detach()
        grad_ori_style_std_block1 = ori_style_std_block1.grad.detach()
        if gradient_gates is not None and len(gradient_gates) > 0:
          gate = gradient_gates[0].to(device)
          grad_ori_style_mean_block1 = grad_ori_style_mean_block1 * gate
          grad_ori_style_std_block1 = grad_ori_style_std_block1 * gate
        index = torch.randint(0, len(epsilon_list), (1, ))[0]
        epsilon = scaled_epsilon_list[index]
        adv_style_mean_block1 = fgsm_attack(ori_style_mean_block1, epsilon, grad_ori_style_mean_block1)
        adv_style_std_block1 = fgsm_attack(ori_style_std_block1, epsilon, grad_ori_style_std_block1)

    # add zero_grad
    self.feature.zero_grad()
    self.classifier.zero_grad()

    if('2' in blocklist and scaled_epsilon_list[1] != 0):
      # forward block1
      x_ori_block1 = self.feature.forward_block1(x_ori)
      # update adv_block1
      if self.semantic_drift_control_enabled:
        x_adv_block1 = self._apply_deterministic_style(
          x_ori_block1,
          adv_style_mean_block1,
          adv_style_std_block1,
        )
      else:
        x_adv_block1 = changeNewAdvStyle(x_ori_block1, adv_style_mean_block1, adv_style_std_block1, p_thred=0)
      # forward block2
      if use_progressive:
        x_ori_block2_base = self.feature.forward_block2(x_adv_block1)
      else:
        x_ori_block2_base = self.feature.forward_block2(x_ori_block1)
      x_ori_block2 = x_ori_block2_base
      # calculate mean and std
      feat_size_block2 = x_ori_block2.size()
      ori_style_mean_block2, ori_style_std_block2 = calc_mean_std(x_ori_block2)
      # set them as learnable parameters
      ori_style_mean_block2  = torch.nn.Parameter(ori_style_mean_block2)
      ori_style_std_block2 = torch.nn.Parameter(ori_style_std_block2)
      ori_style_mean_block2.requires_grad_()
      ori_style_std_block2.requires_grad_()
      # contain ori_style_mean_block1 in the graph 
      x_normalized_block2 = (x_ori_block2 - ori_style_mean_block2.detach().expand(feat_size_block2)) / ori_style_std_block2.detach().expand(feat_size_block2)
      x_ori_block2 = x_normalized_block2 * ori_style_std_block2.expand(feat_size_block2) + ori_style_mean_block2.expand(feat_size_block2)
      # pass the rest model
      x_ori_block3 = self.feature.forward_block3(x_ori_block2)
      x_ori_block4 = self.feature.forward_block4(x_ori_block3)
      x_ori_fea = self.feature.forward_rest(x_ori_block4)
      x_ori_output = self.classifier.forward(x_ori_fea)
      # calculate initial pred, loss and acc
      ori_pred = x_ori_output.max(1, keepdim=True)[1]
      ori_loss = self.loss_fn(x_ori_output, y_ori)
      ori_acc = (ori_pred == y_ori).type(torch.float).sum().item() / y_ori.size()[0]
      if self.semantic_drift_control_enabled:
        task_grad_mean, task_grad_std = torch.autograd.grad(
          ori_loss,
          (ori_style_mean_block2, ori_style_std_block2),
          retain_graph=True,
        )
        semantic_loss_stage, semantic_grad_mean, semantic_grad_std = self._semantic_attack_stage_gradients(
          x_ori_fea,
          ori_style_mean_block2,
          ori_style_std_block2,
          semantic_global_y_episode,
          semantic_samples_per_class,
        )
        epsilon = float(epsilon_list[1])
        lambda_before = semantic_lambda.detach()
        adv_style_mean_block2, mean_stats = semantic_style_candidate(
          ori_style_mean_block2.detach(), task_grad_mean.detach(), semantic_grad_mean, epsilon, False
        )
        adv_style_std_block2, std_stats = semantic_style_candidate(
          ori_style_std_block2.detach(), task_grad_std.detach(), semantic_grad_std, epsilon, True
        )
        with torch.no_grad():
          post_block2 = self._apply_deterministic_style(
            x_ori_block2_base.detach(), adv_style_mean_block2, adv_style_std_block2
          )
          post_block3 = self.feature.forward_block3(post_block2)
          post_block4 = self.feature.forward_block4(post_block3)
          post_fea = self.feature.forward_rest(post_block4)
          post_semantic_losses = self._compute_semantic_anchor_loss(
            self.fc(post_fea),
            semantic_global_y_episode,
            semantic_samples_per_class,
            detach_visual=True,
            freeze_projector_for_attack=True,
            reduction='none',
          )
          post_semantic_loss = post_semantic_losses.mean()
          diagnostics = self._semantic_drift_diagnostics(
            post_semantic_losses,
            semantic_clean_loss_per_sample,
            semantic_samples_per_class,
          )
          stage_drift = diagnostics['mean']
        semantic_lambda = update_dual_state(
          semantic_lambda, stage_drift, self.semantic_drift_config
        )
        record_semantic_stage(
          2, post_semantic_loss, diagnostics, lambda_before, mean_stats, std_stats
        )
      else:
        self.feature.zero_grad()
        self.classifier.zero_grad()
        ori_loss.backward()
        grad_ori_style_mean_block2 = ori_style_mean_block2.grad.detach()
        grad_ori_style_std_block2 = ori_style_std_block2.grad.detach()
        if gradient_gates is not None and len(gradient_gates) > 1:
          gate = gradient_gates[1].to(device)
          grad_ori_style_mean_block2 = grad_ori_style_mean_block2 * gate
          grad_ori_style_std_block2 = grad_ori_style_std_block2 * gate
        index = torch.randint(0, len(epsilon_list), (1, ))[0]
        epsilon = scaled_epsilon_list[index]
        adv_style_mean_block2 = fgsm_attack(ori_style_mean_block2, epsilon, grad_ori_style_mean_block2)
        adv_style_std_block2 = fgsm_attack(ori_style_std_block2, epsilon, grad_ori_style_std_block2)

    # add zero_grad
    self.feature.zero_grad()
    self.classifier.zero_grad()

    if('3' in blocklist and scaled_epsilon_list[2] != 0):
      # forward block1, block2, block3
      x_ori_block1 = self.feature.forward_block1(x_ori)
      if self.semantic_drift_control_enabled:
        x_adv_block1 = self._apply_deterministic_style(
          x_ori_block1,
          adv_style_mean_block1,
          adv_style_std_block1,
        )
      else:
        x_adv_block1 = changeNewAdvStyle(x_ori_block1, adv_style_mean_block1, adv_style_std_block1, p_thred=0)
      if use_progressive:
        x_ori_block2_base = self.feature.forward_block2(x_adv_block1)
      else:
        x_ori_block2_base = self.feature.forward_block2(x_ori_block1)
      x_ori_block2 = x_ori_block2_base
      if self.semantic_drift_control_enabled:
        x_adv_block2 = self._apply_deterministic_style(
          x_ori_block2_base,
          adv_style_mean_block2,
          adv_style_std_block2,
        )
      else:
        x_adv_block2 = changeNewAdvStyle(x_ori_block2, adv_style_mean_block2, adv_style_std_block2, p_thred=0)
      if use_progressive:
        x_ori_block3_base = self.feature.forward_block3(x_adv_block2)
      else:
        x_ori_block3_base = self.feature.forward_block3(x_ori_block2)
      x_ori_block3 = x_ori_block3_base
      # calculate mean and std
      feat_size_block3 = x_ori_block3.size()
      ori_style_mean_block3, ori_style_std_block3 = calc_mean_std(x_ori_block3)
      # set them as learnable parameters
      ori_style_mean_block3  = torch.nn.Parameter(ori_style_mean_block3)
      ori_style_std_block3 = torch.nn.Parameter(ori_style_std_block3)
      ori_style_mean_block3.requires_grad_()
      ori_style_std_block3.requires_grad_()
      # contain ori_style_mean_block3 in the graph 
      x_normalized_block3 = (x_ori_block3 - ori_style_mean_block3.detach().expand(feat_size_block3)) / ori_style_std_block3.detach().expand(feat_size_block3)
      x_ori_block3 = x_normalized_block3 * ori_style_std_block3.expand(feat_size_block3) + ori_style_mean_block3.expand(feat_size_block3)
      # pass the rest model
      x_ori_block4 = self.feature.forward_block4(x_ori_block3)
      x_ori_fea = self.feature.forward_rest(x_ori_block4)
      x_ori_output = self.classifier.forward(x_ori_fea)
      # calculate initial pred, loss and acc
      ori_pred = x_ori_output.max(1, keepdim=True)[1]
      ori_loss = self.loss_fn(x_ori_output, y_ori)
      ori_acc = (ori_pred == y_ori).type(torch.float).sum().item() / y_ori.size()[0]
      if self.semantic_drift_control_enabled:
        task_grad_mean, task_grad_std = torch.autograd.grad(
          ori_loss,
          (ori_style_mean_block3, ori_style_std_block3),
          retain_graph=True,
        )
        semantic_loss_stage, semantic_grad_mean, semantic_grad_std = self._semantic_attack_stage_gradients(
          x_ori_fea,
          ori_style_mean_block3,
          ori_style_std_block3,
          semantic_global_y_episode,
          semantic_samples_per_class,
        )
        epsilon = float(epsilon_list[2])
        lambda_before = semantic_lambda.detach()
        adv_style_mean_block3, mean_stats = semantic_style_candidate(
          ori_style_mean_block3.detach(), task_grad_mean.detach(), semantic_grad_mean, epsilon, False
        )
        adv_style_std_block3, std_stats = semantic_style_candidate(
          ori_style_std_block3.detach(), task_grad_std.detach(), semantic_grad_std, epsilon, True
        )
        with torch.no_grad():
          post_block3 = self._apply_deterministic_style(
            x_ori_block3_base.detach(), adv_style_mean_block3, adv_style_std_block3
          )
          post_block4 = self.feature.forward_block4(post_block3)
          post_fea = self.feature.forward_rest(post_block4)
          post_semantic_losses = self._compute_semantic_anchor_loss(
            self.fc(post_fea),
            semantic_global_y_episode,
            semantic_samples_per_class,
            detach_visual=True,
            freeze_projector_for_attack=True,
            reduction='none',
          )
          post_semantic_loss = post_semantic_losses.mean()
          diagnostics = self._semantic_drift_diagnostics(
            post_semantic_losses,
            semantic_clean_loss_per_sample,
            semantic_samples_per_class,
          )
          stage_drift = diagnostics['mean']
        semantic_lambda = update_dual_state(
          semantic_lambda, stage_drift, self.semantic_drift_config
        )
        record_semantic_stage(
          3, post_semantic_loss, diagnostics, lambda_before, mean_stats, std_stats
        )
      else:
        self.feature.zero_grad()
        self.classifier.zero_grad()
        ori_loss.backward()
        grad_ori_style_mean_block3 = ori_style_mean_block3.grad.detach()
        grad_ori_style_std_block3 = ori_style_std_block3.grad.detach()
        if gradient_gates is not None and len(gradient_gates) > 2:
          gate = gradient_gates[2].to(device)
          grad_ori_style_mean_block3 = grad_ori_style_mean_block3 * gate
          grad_ori_style_std_block3 = grad_ori_style_std_block3 * gate
        index = torch.randint(0, len(epsilon_list), (1, ))[0]
        epsilon = scaled_epsilon_list[index]
        adv_style_mean_block3 = fgsm_attack(ori_style_mean_block3, epsilon, grad_ori_style_mean_block3)
        adv_style_std_block3 = fgsm_attack(ori_style_std_block3, epsilon, grad_ori_style_std_block3)

    return adv_style_mean_block1, adv_style_std_block1, adv_style_mean_block2, adv_style_std_block2, adv_style_mean_block3, adv_style_std_block3 
    
  
  def set_statues_of_modules(self, flag):
    if(flag=='eval'):
      self.feature.eval()
      self.fc.eval()
      self.gnn.eval()
      self.classifier.eval()
      if self.semantic_anchor is not None:
        self.semantic_anchor.eval()
    elif(flag=='train'):
      self.feature.train()
      self.fc.train()
      self.gnn.train()
      self.classifier.train()
      if self.semantic_anchor is not None:
        self.semantic_anchor.train()
    return

  def forward_global_classifier(self, x):
    """Forward images through the shared feature extractor and global classifier."""
    x = x.to(self.device, non_blocking=True)
    features = self.feature.forward(x)
    return self.classifier.forward(features)

  def forward_global_features(self, x):
    """Forward images through the shared feature extractor only."""
    x = x.to(self.device, non_blocking=True)
    return self.feature.forward(x)

  def adjust_target_ssl_weight(self, base_weight, target_stats):
    """Scale target SSL weight based on how many reliable samples survived filtering."""
    if self.target_ssl_mode == "feature_consistency":
      self.last_target_ssl_weight_factor = 1.0
      self.last_target_ssl_keep_ratio = float(target_stats.get("selected_ratio", 1.0))
      self.last_target_ssl_selected_confidence = 0.0
      return float(base_weight), 1.0

    selected_ratio = float(target_stats.get("selected_ratio", 0.0))
    selected_conf = float(target_stats.get("selected_confidence", target_stats.get("pseudo_confidence", 0.0)))
    threshold = min(max(self.target_ssl_confidence_threshold, 0.0), 1.0)

    if selected_ratio <= 0.0 or selected_conf <= 0.0:
      adaptive_factor = 0.0
    else:
      conf_denom = max(1e-6, 1.0 - threshold)
      confidence_factor = max(0.0, min(1.0, (selected_conf - threshold) / conf_denom))
      adaptive_factor = math.sqrt(selected_ratio) * confidence_factor

    adaptive_factor = max(self.target_ssl_weight_floor, min(1.0, adaptive_factor))
    self.last_target_ssl_weight_factor = float(adaptive_factor)
    self.last_target_ssl_keep_ratio = float(selected_ratio)
    self.last_target_ssl_selected_confidence = float(selected_conf)
    return float(base_weight) * adaptive_factor, adaptive_factor

  def compute_target_unlabeled_loss(self, x_weak, x_strong, temperature=0.5):
    """
    Compute confidence-aware target-domain consistency with feature and
    pseudo-prototype refinement terms.
    """
    temperature = max(float(temperature), 1e-6)
    weak_features = self.forward_global_features(x_weak)
    strong_features = self.forward_global_features(x_strong)

    if self.target_ssl_mode == "feature_consistency":
      weak_norm = F.normalize(weak_features.detach(), dim=1)
      strong_norm = F.normalize(strong_features, dim=1)
      view_agreement = (weak_norm * strong_norm).sum(dim=1)
      feature_loss = (1.0 - view_agreement).mean()
      zero = feature_loss.detach().new_zeros(())
      stats = {
        "pseudo_confidence": 0.0,
        "mean_entropy": 0.0,
        "selected_ratio": 1.0,
        "selected_confidence": 0.0,
        "selected_entropy": 0.0,
        "view_agreement": float(view_agreement.detach().mean().cpu()),
        "selected_view_agreement": float(view_agreement.detach().mean().cpu()),
        "prob_loss": float(zero.cpu()),
        "feature_loss": float(feature_loss.detach().cpu()),
        "prototype_loss": float(zero.cpu()),
      }
      return feature_loss, stats

    weak_logits = self.classifier.forward(weak_features)
    strong_logits = self.classifier.forward(strong_features)
    strong_log_probs = F.log_softmax(strong_logits, dim=1)
    strong_norm = F.normalize(strong_features, dim=1)
    log_class_count = math.log(max(weak_logits.size(1), 2))

    with torch.no_grad():
      pseudo_probs = torch.softmax(weak_logits / temperature, dim=1)
      top2_probs = pseudo_probs.topk(k=min(2, pseudo_probs.size(1)), dim=1).values
      pseudo_conf_values = top2_probs[:, 0]
      pseudo_conf = pseudo_conf_values.mean()
      pseudo_labels = pseudo_probs.argmax(dim=1)
      pseudo_entropy = -(pseudo_probs * torch.log(pseudo_probs.clamp_min(1e-8))).sum(dim=1) / log_class_count
      mean_entropy = pseudo_entropy.mean()
      weak_norm = F.normalize(weak_features, dim=1)
      view_agreement = (weak_norm * strong_norm).sum(dim=1)
      mean_view = view_agreement.mean()

      conf_mask = pseudo_conf_values >= self.target_ssl_confidence_threshold
      entropy_mask = pseudo_entropy <= self.target_ssl_max_entropy
      view_mask = view_agreement >= self.target_ssl_view_threshold
      keep_mask = conf_mask & entropy_mask & view_mask
      selected_ratio = keep_mask.float().mean()
      selected_conf = pseudo_conf_values[keep_mask].mean() if keep_mask.any() else pseudo_conf_values.new_tensor(0.0)
      selected_entropy = pseudo_entropy[keep_mask].mean() if keep_mask.any() else pseudo_entropy.new_tensor(1.0)
      selected_view = view_agreement[keep_mask].mean() if keep_mask.any() else view_agreement.new_tensor(0.0)
      weak_norm = weak_norm.detach()
      pseudo_probs = pseudo_probs.detach()

    per_sample_prob_loss = -(pseudo_probs * strong_log_probs).sum(dim=1)
    per_sample_feature_loss = 1.0 - (weak_norm * strong_norm).sum(dim=1)

    if keep_mask.any():
      prob_loss = per_sample_prob_loss[keep_mask].mean()
      feature_loss = per_sample_feature_loss[keep_mask].mean()

      masked_weak = weak_norm[keep_mask]
      masked_strong = strong_norm[keep_mask]
      masked_labels = pseudo_labels[keep_mask]
      prototype_targets = torch.zeros_like(masked_weak)
      for label in masked_labels.unique(sorted=True):
        label_mask = masked_labels == label
        proto = F.normalize(masked_weak[label_mask].mean(dim=0, keepdim=True), dim=-1)
        prototype_targets[label_mask] = proto.expand(label_mask.sum(), -1)
      prototype_loss = 1.0 - (masked_strong * prototype_targets).sum(dim=1).mean()
      loss = (
        prob_loss
        + self.target_ssl_feature_weight * feature_loss
        + self.target_ssl_prototype_weight * prototype_loss
      )
    else:
      zero = strong_logits.new_zeros(())
      prob_loss = zero
      feature_loss = zero
      prototype_loss = zero
      loss = zero

    stats = {
      "pseudo_confidence": float(pseudo_conf.detach().cpu()),
      "mean_entropy": float(mean_entropy.detach().cpu()),
      "selected_ratio": float(selected_ratio.detach().cpu()),
      "selected_confidence": float(selected_conf.detach().cpu()),
      "selected_entropy": float(selected_entropy.detach().cpu()),
      "view_agreement": float(mean_view.detach().cpu()),
      "selected_view_agreement": float(selected_view.detach().cpu()),
      "prob_loss": float(prob_loss.detach().cpu()),
      "feature_loss": float(feature_loss.detach().cpu()),
      "prototype_loss": float(prototype_loss.detach().cpu()),
    }
    return loss, stats
   

  def set_forward_loss_StyAdv(self, x_ori, global_y, epsilon_list):
    ##################################################################
    # 0. first cp x_adv from x_ori
    x_adv = x_ori
    
    ##################################################################
    # 0.5. 璁＄畻鏂囨湰寮曞鍙傛暟 (濡傛灉鍚敤)
    epsilon_scales = None
    gradient_gates = None
    
    if self.text_guide_epsilon or self.text_guide_gradient:
      # 鑾峰彇鏍锋湰鏁伴噺淇℃伅
      x_size = x_ori.size()
      samples_per_class = x_size[1]  # n_support + n_query
      
      # global_y: [n_way * samples_per_class] 鎴?[n_way, samples_per_class]
      global_y_flat = global_y.view(-1)
      
      # 璁＄畻鏂囨湰寮曞
      eps_scales, grad_gates = self._compute_text_guidance(global_y_flat, samples_per_class)
      
      if self.text_guide_epsilon:
        epsilon_scales = eps_scales
      if self.text_guide_gradient:
        gradient_gates = grad_gates
    
    use_style_adv = not getattr(self, 'disable_style_adv_generator', False)
    if use_style_adv:
      ##################################################################
      # 1. styleAdv
      self.set_statues_of_modules('eval') 

      semantic_clean_loss_for_attack = None
      semantic_clean_loss_per_sample_for_attack = None
      semantic_global_y_episode_for_attack = None
      semantic_samples_per_class_for_attack = None
      if self.semantic_drift_control_enabled:
        attack_x_size = x_ori.size()
        semantic_samples_per_class_for_attack = attack_x_size[1]
        semantic_global_y_episode_for_attack = global_y.to(self.device).view(
          attack_x_size[0], semantic_samples_per_class_for_attack
        )
        clean_x = x_ori.to(self.device, non_blocking=True).view(
          attack_x_size[0] * attack_x_size[1],
          attack_x_size[2],
          attack_x_size[3],
          attack_x_size[4],
        )
        with torch.no_grad():
          clean_block1 = self.feature.forward_block1(clean_x)
          clean_block2 = self.feature.forward_block2(clean_block1)
          clean_block3 = self.feature.forward_block3(clean_block2)
          clean_block4 = self.feature.forward_block4(clean_block3)
          clean_fea = self.feature.forward_rest(clean_block4)
          semantic_clean_loss_per_sample_for_attack = self._compute_semantic_anchor_loss(
            self.fc(clean_fea),
            semantic_global_y_episode_for_attack,
            semantic_samples_per_class_for_attack,
            detach_visual=True,
            freeze_projector_for_attack=True,
            reduction='none',
          ).detach()
          semantic_clean_loss_for_attack = semantic_clean_loss_per_sample_for_attack.mean()

      adv_style_mean_block1, adv_style_std_block1, adv_style_mean_block2, adv_style_std_block2, adv_style_mean_block3, adv_style_std_block3 = self.adversarial_attack_Incre(
        x_ori, global_y, epsilon_list, 
        epsilon_scales=epsilon_scales, 
        gradient_gates=gradient_gates,
        attack_mode=self.style_attack_mode,
        semantic_clean_loss=semantic_clean_loss_for_attack,
        semantic_clean_loss_per_sample=semantic_clean_loss_per_sample_for_attack,
        semantic_global_y_episode=semantic_global_y_episode_for_attack,
        semantic_samples_per_class=semantic_samples_per_class_for_attack,
      )
 
      self.feature.zero_grad()
      self.fc.zero_grad()
      self.classifier.zero_grad()
      self.gnn.zero_grad()

    #################################################################
    # 2. forward and get loss
    self.set_statues_of_modules('train')

    # define y_query for FSL
    y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query))
    y_query = y_query.to(self.device, non_blocking=True)

    # forward x_ori 
    x_ori = x_ori.to(self.device, non_blocking=True)
    x_size = x_ori.size()
    samples_per_class = x_size[1]
    x_ori = x_ori.view(x_size[0]*x_size[1], x_size[2], x_size[3], x_size[4])
    global_y = global_y.to(self.device, non_blocking=True)
    global_y_episode = global_y.view(x_size[0], samples_per_class)
    flat_global_y = global_y_episode.contiguous().view(-1)
    x_ori_block1 = self.feature.forward_block1(x_ori)
    x_ori_block2 = self.feature.forward_block2(x_ori_block1)
    x_ori_block3 = self.feature.forward_block3(x_ori_block2)
    x_ori_block4 = self.feature.forward_block4(x_ori_block3)
    x_ori_fea = self.feature.forward_rest(x_ori_block4)

    # ori cls global loss    
    scores_cls_ori = self.classifier.forward(x_ori_fea)
    loss_cls_ori = self.loss_fn(scores_cls_ori.float(), flat_global_y.long())
    acc_cls_ori = ( scores_cls_ori.max(1, keepdim=True)[1]  == flat_global_y ).type(torch.float).sum().item() / flat_global_y.size()[0]

    # ori FSL scores and losses
    x_ori_z = self.fc(x_ori_fea)
    x_ori_z = x_ori_z.view(self.n_way, -1, x_ori_z.size(1))

    # Proposed semantic anchor: clean outer path updates only P.
    loss_semantic_anchor = self._compute_semantic_anchor_loss(
      x_ori_z.reshape(-1, x_ori_z.size(-1)),
      global_y_episode,
      samples_per_class,
      detach_visual=True,
    )
    self.last_semantic_anchor_loss = float(loss_semantic_anchor.detach().cpu())
    
    # [v2.1 鏂规B] 璁粌鏃朵篃搴旂敤text calibration锛岃projection灞傚涔?
    loss_text_align = torch.tensor(0.0, device=self.device)
    if self.text_calibration_weight > 0 and hasattr(self, 'text_to_visual_proj'):
      # 鑾峰彇褰撳墠episode鐨勭被鍚?
      class_ids = global_y_episode[:, 0].long().tolist()
      class_names = self._get_class_names_for_ids(class_ids)
      
      # 搴旂敤calibration (甯︽搴?
      x_ori_z, loss_text_align = self._apply_text_calibration_v2_train(x_ori_z, class_names)
    
    x_ori_z_stack = self._build_query_episode_stack(x_ori_z)
    assert(x_ori_z_stack.size(1) == self.n_way*(self.n_support + 1))
    scores_fsl_ori = self.forward_gnn(x_ori_z_stack).float()
    loss_fsl_ori = (
      self.loss_fn(scores_fsl_ori, y_query.long())
      + 0.1 * loss_text_align
      + self.semantic_anchor_weight * loss_semantic_anchor
    )

    if not use_style_adv:
      zero = loss_fsl_ori.new_zeros(())
      return (
        scores_fsl_ori,
        loss_fsl_ori,
        scores_cls_ori,
        loss_cls_ori,
        scores_fsl_ori.detach(),
        zero,
        scores_cls_ori.detach(),
        zero,
      )

    # forward x_adv
    x_adv = x_adv.to(self.device, non_blocking=True)
    x_adv = x_adv.view(x_size[0]*x_size[1], x_size[2], x_size[3], x_size[4])
    x_adv_block1 = self.feature.forward_block1(x_adv)

    if self.semantic_drift_control_enabled:
      x_adv_block1_newStyle = self._apply_deterministic_style(
        x_adv_block1, adv_style_mean_block1, adv_style_std_block1
      )
    else:
      x_adv_block1_newStyle = changeNewAdvStyle(
        x_adv_block1, adv_style_mean_block1, adv_style_std_block1, p_thred=P_THRED
      )
    if self.style_attack_mode == 'independent':
      x_adv_block2_input = self.feature.forward_block2(x_adv_block1)
      if self.semantic_drift_control_enabled:
        x_adv_block2_newStyle = self._apply_deterministic_style(
          x_adv_block2_input, adv_style_mean_block2, adv_style_std_block2
        )
      else:
        x_adv_block2_newStyle = changeNewAdvStyle(
          x_adv_block2_input, adv_style_mean_block2, adv_style_std_block2, p_thred=P_THRED
        )
      x_adv_block3_input = self.feature.forward_block3(x_adv_block2_input)
      if self.semantic_drift_control_enabled:
        x_adv_block3_newStyle = self._apply_deterministic_style(
          x_adv_block3_input, adv_style_mean_block3, adv_style_std_block3
        )
      else:
        x_adv_block3_newStyle = changeNewAdvStyle(
          x_adv_block3_input, adv_style_mean_block3, adv_style_std_block3, p_thred=P_THRED
        )
    else:
      x_adv_block2 = self.feature.forward_block2(x_adv_block1_newStyle)
      if self.semantic_drift_control_enabled:
        x_adv_block2_newStyle = self._apply_deterministic_style(
          x_adv_block2, adv_style_mean_block2, adv_style_std_block2
        )
      else:
        x_adv_block2_newStyle = changeNewAdvStyle(
          x_adv_block2, adv_style_mean_block2, adv_style_std_block2, p_thred=P_THRED
        )
      x_adv_block3 = self.feature.forward_block3(x_adv_block2_newStyle)
      if self.semantic_drift_control_enabled:
        x_adv_block3_newStyle = self._apply_deterministic_style(
          x_adv_block3, adv_style_mean_block3, adv_style_std_block3
        )
      else:
        x_adv_block3_newStyle = changeNewAdvStyle(
          x_adv_block3, adv_style_mean_block3, adv_style_std_block3, p_thred=P_THRED
        )
    x_adv_block4 = self.feature.forward_block4(x_adv_block3_newStyle)
    x_adv_fea = self.feature.forward_rest(x_adv_block4)
   
    # adv cls gloabl loss
    scores_cls_adv = self.classifier.forward(x_adv_fea)
    loss_cls_adv = self.loss_fn(scores_cls_adv.float(), flat_global_y.long())
    acc_cls_adv = ( scores_cls_adv.max(1, keepdim=True)[1]  == flat_global_y ).type(torch.float).sum().item() / flat_global_y.size()[0]

    # adv FSL scores and losses
    x_adv_z = self.fc(x_adv_fea)
    x_adv_z = x_adv_z.view(self.n_way, -1, x_adv_z.size(1))
    x_adv_z_stack = self._build_query_episode_stack(x_adv_z)
    assert(x_adv_z_stack.size(1) == self.n_way*(self.n_support + 1))
    scores_fsl_adv = self.forward_gnn(x_adv_z_stack).float()
    loss_fsl_adv = self.loss_fn(scores_fsl_adv, y_query.long())

    #print('scores_fsl_adv:', scores_fsl_adv.mean(), 'loss_fsl_adv:', loss_fsl_adv, 'scores_cls_adv:', scores_cls_adv.mean(), 'loss_cls_adv:', loss_cls_adv)
    return scores_fsl_ori, loss_fsl_ori, scores_cls_ori, loss_cls_ori, scores_fsl_adv, loss_fsl_adv, scores_cls_adv, loss_cls_adv
