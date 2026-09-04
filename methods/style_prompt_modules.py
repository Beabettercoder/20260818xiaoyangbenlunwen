import warnings
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
  import clip  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
  clip = None


class TextPromptEncoder(nn.Module):
  """Lightweight CLIP text encoder that produces class embeddings and a task context."""

  def __init__(self, clip_model_name: str = "ViT-B/32", device: Optional[torch.device] = None):
    super().__init__()
    if clip is None:
      raise ImportError("`clip` package is required for TextPromptEncoder")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = clip.load(clip_model_name, device=device)
    model.float()
    model.eval()
    for p in model.parameters():
      p.requires_grad_(False)
    self.clip_model = model
    self.text_dim = model.text_projection.shape[1] if model.text_projection.dim() == 2 else model.text_projection.shape[0]
    self.device = device

  def encode_class_prompts(self, class_prompts: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode per-class prompts into text embeddings and aggregate into a task context."""
    if len(class_prompts) == 0:
      raise ValueError("class_prompts is empty; cannot encode text prompts")
    tokens = clip.tokenize(list(class_prompts), truncate=True).to(self.device)
    with torch.no_grad():
      text_embeddings = self.clip_model.encode_text(tokens).float()
      text_embeddings = F.normalize(text_embeddings, dim=-1)
    z_text = text_embeddings.mean(dim=0)
    z_text = F.normalize(z_text, dim=-1)
    return text_embeddings, z_text

  def encode_class_prompts_ensemble(
    self, 
    class_names: Sequence[str], 
    templates: Sequence[str]
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    [Step1: PromptEnsemble] 
    多模板集成编码：对每个类别使用多个模板，取平均作为类别嵌入。
    
    原理参考 CLIP 论文的 ImageNet 零样本分类:
      "We ensemble over the text embeddings of the 80 prompt templates"
    
    Args:
      class_names: 类别名称列表 [n_way]
      templates: 模板列表，使用 {name} 作为占位符
    
    Returns:
      class_embeddings: [n_way, text_dim] 每个类的集成嵌入
      z_text: [text_dim] 任务级别的上下文向量
    """
    if len(class_names) == 0:
      raise ValueError("class_names is empty")
    if len(templates) == 0:
      raise ValueError("templates is empty")
    
    n_classes = len(class_names)
    n_templates = len(templates)
    
    # 为每个类生成所有模板的 prompt
    all_prompts = []
    for cls_name in class_names:
      for template in templates:
        all_prompts.append(template.format(name=cls_name))
    
    # 批量编码所有 prompts
    tokens = clip.tokenize(all_prompts, truncate=True).to(self.device)
    with torch.no_grad():
      all_embeddings = self.clip_model.encode_text(tokens).float()  # [n_classes * n_templates, dim]
      all_embeddings = F.normalize(all_embeddings, dim=-1)
    
    # reshape 为 [n_classes, n_templates, dim]
    all_embeddings = all_embeddings.view(n_classes, n_templates, -1)
    
    # 对每个类的多模板取平均 (Prompt Ensemble)
    class_embeddings = all_embeddings.mean(dim=1)  # [n_classes, dim]
    class_embeddings = F.normalize(class_embeddings, dim=-1)
    
    # 任务级上下文: 所有类的平均
    z_text = class_embeddings.mean(dim=0)  # [dim]
    z_text = F.normalize(z_text, dim=-1)
    
    return class_embeddings, z_text

  def encode_class_prompts_ensemble_with_stats(
    self,
    class_names: Sequence[str],
    templates: Sequence[str],
  ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Encode prompt ensembles and return simple agreement statistics."""
    if len(class_names) == 0:
      raise ValueError("class_names is empty")
    if len(templates) == 0:
      raise ValueError("templates is empty")

    n_classes = len(class_names)
    n_templates = len(templates)

    all_prompts = []
    for cls_name in class_names:
      for template in templates:
        all_prompts.append(template.format(name=cls_name))

    tokens = clip.tokenize(all_prompts, truncate=True).to(self.device)
    with torch.no_grad():
      all_embeddings = self.clip_model.encode_text(tokens).float()
      all_embeddings = F.normalize(all_embeddings, dim=-1)

    all_embeddings = all_embeddings.view(n_classes, n_templates, -1)
    class_embeddings = F.normalize(all_embeddings.mean(dim=1), dim=-1)
    z_text = F.normalize(class_embeddings.mean(dim=0), dim=-1)

    class_centers = class_embeddings.unsqueeze(1).expand_as(all_embeddings)
    class_agreement = F.cosine_similarity(all_embeddings, class_centers, dim=-1).mean(dim=1)
    task_agreement = class_agreement.mean()
    stats = {
      "task_agreement": float(task_agreement.item()),
      "class_agreement": class_agreement.detach(),
    }
    return class_embeddings, z_text, stats


class StylePromptGenerator(nn.Module):
  """Maps text context to style control codes."""

  def __init__(self, in_dim: int, out_dim: int, hidden_dim: Optional[int] = None):
    super().__init__()
    hidden = hidden_dim or max(in_dim, out_dim)
    self.net = nn.Sequential(
      nn.Linear(in_dim, hidden),
      nn.ReLU(inplace=True),
      nn.Linear(hidden, out_dim),
    )

  def forward(self, z_text: torch.Tensor) -> torch.Tensor:
    if z_text.dim() == 1:
      z_text = z_text.unsqueeze(0)
    out = self.net(z_text)
    return out.squeeze(0)


class ChannelGatingNetwork(nn.Module):
  """
  [TG-MSAT Added]
  Generates channel-wise gating vectors (alphas) from a text-derived style prompt.
  These gates modulate the strength of adversarial attacks on a per-channel basis.
  
  [Phase 1 优化] 支持可配置的门控范围 [gate_min, gate_max]
  """
  def __init__(self, style_prompt_dim: int, block_channels: List[int], 
               gate_min: float = 0.0, gate_max: float = 1.0):
    """
    Args:
      style_prompt_dim: The dimension of the input style prompt vector.
      block_channels: A list of integers specifying the number of channels for each ResNet block to be gated.
      gate_min: 门控最小值 (默认0.0)
      gate_max: 门控最大值 (默认1.0)
    """
    super().__init__()
    self.block_channels = block_channels
    self.gate_min = gate_min
    self.gate_max = gate_max
    
    # Create a gate generator for each block (without Sigmoid, we apply it manually)
    self.gate_generators = nn.ModuleList()
    for num_channels in self.block_channels:
        self.gate_generators.append(
            nn.Linear(style_prompt_dim, num_channels)
        )

  def forward(self, style_prompt: torch.Tensor) -> List[torch.Tensor]:
    """
    Takes a style prompt and returns a list of gating tensors.
    
    Args:
      style_prompt: A tensor of shape [style_prompt_dim] derived from text features.
    
    Returns:
      A list of gating tensors [alpha1, alpha2, ...], where each alpha has a shape
      of (1, C, 1, 1) for its corresponding block's channel count C.
    """
    if style_prompt.dim() == 1:
        style_prompt = style_prompt.unsqueeze(0)

    alpha_list = []
    for generator in self.gate_generators:
        logits = generator(style_prompt)
        # 可配置的门控范围: [gate_min, gate_max]
        gate_range = self.gate_max - self.gate_min
        alpha = self.gate_min + gate_range * torch.sigmoid(logits)
        # Output shape: (1, num_channels) -> (1, num_channels, 1, 1) for broadcasting
        alpha = alpha.view(1, -1, 1, 1)
        alpha_list.append(alpha)
        
    return alpha_list


class ClipAlignHead(nn.Module):
  """Aligns visual prototypes with text context via a shared projection."""

  def __init__(self, d_vis: int, d_text: int, d_align: int):
    super().__init__()
    self.proj_vis = nn.Linear(d_vis, d_align)
    self.proj_text = nn.Linear(d_text, d_align)

  def forward(self, prototypes: torch.Tensor, z_text: torch.Tensor) -> torch.Tensor:
    """
    Args:
      prototypes: [C, d_vis] visual prototypes.
      z_text: [d_text] or [1, d_text] text context.
    Returns:
      Scalar loss = 1 - mean cosine similarity.
    """
    if z_text.dim() == 1:
      z_text = z_text.unsqueeze(0)
    v = F.normalize(self.proj_vis(prototypes), dim=-1)
    t = F.normalize(self.proj_text(z_text), dim=-1)
    if t.size(0) == 1:
      t = t.expand(v.size(0), -1)
    cos = (v * t).sum(dim=-1)
    return 1.0 - cos.mean()
