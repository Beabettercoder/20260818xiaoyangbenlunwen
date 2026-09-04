import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

from options import parse_args
from utils.PSG import PseudoSampleGenerator
from methods.backbone_multiblock import model_dict
from methods.StyleAdv_RN_GNN import StyleAdvGNN
from data.datamgr import SetDataManager, get_few_shot_datamgr
from config_datasets import get_all_datasets
from utils.distributed_utils import cleanup_distributed, is_dist_avail_and_initialized, is_main_process, setup_distributed
from utils.progress_utils import progress_range
from utils.path_utils import validate_dataset_root


FINETUNE_LR = 0.005
N_PSEUDO = 75
ALL_DATASETS = set(get_all_datasets())


def resolve_target_dataset(params):
  """Use --dataset when provided; fall back to legacy --testset."""
  dataset = params.dataset
  if dataset:
    return dataset
  if getattr(params, "testset", None):
    return params.testset
  raise ValueError("Please specify --dataset for metatest_StyleAdv_RN.py")


def build_target_loader(params, image_size, iter_num, n_query):
  target_dataset = resolve_target_dataset(params)
  if target_dataset not in ALL_DATASETS:
    raise ValueError(f"Unsupported target dataset: {target_dataset}")
  few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
  loader_kwargs = dict(
    num_workers=params.eval_num_workers,
    pin_memory=bool(getattr(params, "pin_memory", 1)),
    persistent_workers=bool(getattr(params, "persistent_workers", 1)),
    prefetch_factor=params.prefetch_factor,
    sampler_seed=0,
    world_size=getattr(params, "world_size", 1),
    rank=getattr(params, "rank", 0),
  )

  novel_file = os.path.join(params.data_dir, target_dataset, "novel.json")
  if os.path.isfile(novel_file):
    datamgr = SetDataManager(
      image_size,
      n_query=n_query,
      n_way=params.test_n_way,
      n_support=params.n_shot,
      n_eposide=iter_num,
      data_root=params.data_dir,
      dataset_name=target_dataset,
      **loader_kwargs,
    )
    return datamgr.get_data_loader(novel_file, aug=False)

  datamgr = get_few_shot_datamgr(
    target_dataset,
    episodic=True,
    image_size=image_size,
    n_way=params.test_n_way,
    n_support=params.n_shot,
    n_query=n_query,
    n_eposide=iter_num,
    data_root=params.data_dir,
    **loader_kwargs,
  )
  return datamgr.get_data_loader(aug=False)


def load_checkpoint_state(params, device):
  checkpoint_dir = os.path.join(params.save_dir, "checkpoints", params.name)
  if params.save_epoch >= 0:
    ckpt_path = os.path.join(checkpoint_dir, f"{params.save_epoch}.tar")
  else:
    ckpt_path = os.path.join(checkpoint_dir, "best_model.tar")

  if not os.path.isfile(ckpt_path):
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

  print(f"Loading checkpoint: {ckpt_path}")
  checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
  if "state" in checkpoint:
    return checkpoint["state"]
  if "model_state" in checkpoint:
    return checkpoint["model_state"]
  raise KeyError(f"'state' not found in checkpoint: {ckpt_path}")


def _load_state_flexible(model, state_dict):
  """Load state dict while dropping incompatible keys."""
  current = model.state_dict()
  filtered = {}
  dropped = []
  for k, v in state_dict.items():
    if k not in current:
      dropped.append(k)
      continue
    if current[k].shape != v.shape:
      dropped.append(k)
      continue
    filtered[k] = v
  if dropped:
    print(f"[load_state_flexible] dropped {len(dropped)} keys (shape mismatch/unused), e.g., {dropped[:3]}")
  model.load_state_dict(filtered, strict=False)


def finetune_episode(state_dict, episode_batch, episode_labels, params, device):
  n_way = params.test_n_way
  n_support = params.n_shot

  model = StyleAdvGNN(
    model_dict[params.model],
    n_way=n_way,
    n_support=n_support,
    device=device,
    dataset_name=resolve_target_dataset(params),
    data_root=params.data_dir,
    use_style_prompt=getattr(params, "use_style_prompt", 0),
    clip_align_weight=getattr(params, "clip_align_weight", 0.0),
    style_prompt_dim=getattr(params, "style_prompt_dim", None),
    clip_model_name=getattr(params, "clip_model_name", "ViT-B/32"),
    # [消融实验] 超参数
    text_weight_mode=getattr(params, 'text_weight_mode', 'fixed'),
    text_weight_fixed=getattr(params, 'text_weight_fixed', 1.0),
    gate_min=getattr(params, 'gate_min', 0.0),
    gate_max=getattr(params, 'gate_max', 1.0),
    epsilon_scale_min=getattr(params, 'epsilon_scale_min', 0.5),
    epsilon_scale_max=getattr(params, 'epsilon_scale_max', 1.5),
    # [Step1: PromptEnsemble] 多模板集成
    use_prompt_ensemble=bool(getattr(params, 'use_prompt_ensemble', 1)),
    # [Step1.1: Test-time Text Calibration]
    text_calibration_weight=getattr(params, 'text_calibration_weight', 0.0),
    semantic_anchor=getattr(params, 'semantic_anchor', 0),
    semantic_anchor_weight=getattr(params, 'semantic_anchor_weight', 1.0),
    semantic_anchor_temperature=getattr(params, 'semantic_anchor_temperature', 0.07),
    semantic_drift_control=getattr(params, 'semantic_drift_control', 0),
    semantic_risk_beta=getattr(params, 'semantic_risk_beta', 1.0),
    semantic_budget_temperature=getattr(params, 'semantic_budget_temperature', 1.0),
    semantic_drift_threshold=getattr(params, 'semantic_drift_threshold', 0.0),
    semantic_dual_step_size=getattr(params, 'semantic_dual_step_size', 0.1),
    semantic_lambda_max=getattr(params, 'semantic_lambda_max', 10.0),
    semantic_sigma_min=getattr(params, 'semantic_sigma_min', 1e-6),
  ).to(device)
  _load_state_flexible(model, state_dict)

  x = episode_batch.to(device)
  global_y = episode_labels.to(device)
  xs = x[:, :n_support].reshape(-1, *x.size()[2:])

  pseudo_generator = PseudoSampleGenerator(n_way, n_support, N_PSEUDO)
  loss_fn = nn.CrossEntropyLoss().to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=FINETUNE_LR)

  n_query = N_PSEUDO // n_way
  pseudo_targets = torch.from_numpy(np.repeat(range(n_way), n_query)).to(device)
  model.n_query = n_query
  model.train()

  for _ in range(params.finetune_epoch):
    optimizer.zero_grad()
    pseudo_set = pseudo_generator.generate(xs)
    pseudo_set = pseudo_set.view(n_way, n_support + n_query, *x.size()[2:])  # reshape back to episode shape
    scores = model.set_forward(pseudo_set, global_y=global_y)
    loss = loss_fn(scores, pseudo_targets)
    loss.backward()
    optimizer.step()

    del pseudo_set, scores, loss

  try:
    torch.cuda.empty_cache()
  except Exception:
    pass

  model.eval()
  n_query = x.size(1) - n_support
  model.n_query = n_query
  labels = np.repeat(range(n_way), n_query)

  with torch.no_grad():
    scores = model.set_forward(x, global_y=global_y)
    predictions = scores.argmax(dim=1).cpu().numpy()
    acc = (predictions == labels).mean() * 100.0

  del scores, predictions

  try:
    torch.cuda.empty_cache()
  except Exception:
    pass

  return acc


def evaluate(params):
  device = setup_distributed(params)

  iter_num = getattr(params, "n_episodes_test", 1000)
  image_size = 224
  n_query = params.n_query
  target_dataset = resolve_target_dataset(params)
  missing_paths = validate_dataset_root(params.data_dir, [target_dataset], required_splits=('novel',))
  if missing_paths:
    raise FileNotFoundError(
      'Target dataset is not available under --data_dir. '
      f'Check --data_dir={params.data_dir}. Missing:\n{missing_paths}'
    )

  state_dict = load_checkpoint_state(params, device)
  novel_loader = build_target_loader(params, image_size, iter_num, n_query)

  acc_all = []
  start_time = time.perf_counter()

  if is_main_process():
    print(f"Testing {params.name} on {target_dataset} | n_way={params.test_n_way}, n_shot={params.n_shot}")

  iterator = progress_range(novel_loader, desc=f"Metatest {target_dataset}") if is_main_process() else novel_loader
  for task_idx, (x, global_y) in enumerate(iterator):
    acc = finetune_episode(state_dict, x, global_y, params, device)
    acc_all.append(acc)
    if is_main_process():
      print(f"Task {task_idx} : {acc:.2f}%, mean Acc: {np.mean(acc_all):.2f}")

  if is_dist_avail_and_initialized():
    gathered = [None for _ in range(params.world_size)]
    dist.all_gather_object(gathered, acc_all)
    merged_acc = []
    for shard in gathered:
      merged_acc.extend(shard)
    acc_all = merged_acc

  duration = time.perf_counter() - start_time
  acc_all = np.asarray(acc_all)
  acc_mean = acc_all.mean()
  acc_ci = 1.96 * acc_all.std(ddof=1) / np.sqrt(len(acc_all))

  if is_main_process():
    print(f"{len(acc_all)} test iterations on {target_dataset}: Acc = {acc_mean:.2f}% +- {acc_ci:.2f}%")
    print(f"Running time: {duration:.2f}s ({duration/60:.2f} min)")

    results_dir = os.path.join(params.save_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"test_{target_dataset.lower()}.txt")
    with open(results_path, "w") as handle:
      handle.write(f"Test Acc = {acc_mean:.2f} +- {acc_ci:.2f}%\n")
      handle.write(f"Episodes: {len(acc_all)}\n")
      handle.write(f"Running time: {duration:.2f}s\n")

    print(f"Saved results to: {results_path}")
  cleanup_distributed()


if __name__ == "__main__":
  seed = 0
  params = parse_args("test")
  device = setup_distributed(params)
  rank_seed = seed + getattr(params, "rank", 0)
  if is_main_process():
    print(f"set seed = {seed}")
  random.seed(rank_seed)
  np.random.seed(rank_seed)
  torch.manual_seed(rank_seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(rank_seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = True
  # Align semantic-guidance settings with the best preset per shot.
  # Best semantic-guidance preset: fixed=1.0, gate=[0,1], 5-shot->[0.3,1.7], 1-shot->[0.7,1.3]
  params.text_weight_mode = 'fixed'
  params.text_weight_fixed = 1.0
  params.gate_min = 0.0
  params.gate_max = 1.0
  if params.n_shot >= 5:
    params.epsilon_scale_min = 0.3
    params.epsilon_scale_max = 1.7
  else:
    params.epsilon_scale_min = 0.7
    params.epsilon_scale_max = 1.3
  params.finetune_epoch = getattr(params, "finetune_epoch", 10)
  
  # [Step1] 打印关键配置
  if is_main_process():
    print(f"[Test Config] use_prompt_ensemble={getattr(params, 'use_prompt_ensemble', 1)}")
    print(f"[Test Config] epsilon_scale_range=[{params.epsilon_scale_min}, {params.epsilon_scale_max}]")
    print(f"[Test Config] distributed={bool(getattr(params, 'distributed', 0))}, world_size={getattr(params, 'world_size', 1)}, device={device}")

  evaluate(params)
