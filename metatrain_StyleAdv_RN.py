import numpy as np
import torch
import torch.optim
import os
import random 
import warnings

from methods.backbone_multiblock import model_dict
from data.datamgr import SetDataManager, build_unlabeled_dataloader, load_dataset_meta
from methods.StyleAdv_RN_GNN import StyleAdvGNN

from options import parse_args, get_resume_file, load_warmup_state
from test_function_fwt_benchmark import test_bestmodel
# [CLEANED] Removed the broken 'test_function_bscdfsl_benchmark' import
from utils.distributed_utils import (
  barrier,
  cleanup_distributed,
  is_main_process,
  sync_model_state,
  setup_distributed,
  unwrap_model,
)
from utils.progress_utils import progress_range
from utils.plot_utils import plot_curve
from utils.dataset_utils import parse_target_datasets
from utils.path_utils import validate_dataset_root


# Dual writer to write to both console and file simultaneously
class DualWriter:
  """Write output to both stdout and a file object."""
  def __init__(self, file_obj):
    self.file_obj = file_obj
  
  def write(self, msg):
    print(msg, end='')
    self.file_obj.write(msg)
    self.file_obj.flush()
  
  def flush(self):
    self.file_obj.flush()


def _build_meta_from_params(params):
  return dict(
    dataset=params.dataset,
    model=params.model,
    method=params.method,
    train_n_way=params.train_n_way,
    test_n_way=params.test_n_way,
    n_shot=params.n_shot,
    name=params.name,
  )


def _resolve_pairwise_target(params):
  explicit_target = getattr(params, 'target_dataset', '').strip()
  explicit_targets = parse_target_datasets(params.source_dataset, getattr(params, 'target_datasets', ''))

  if explicit_target:
    if explicit_target == params.source_dataset:
      raise ValueError('target_dataset must differ from source_dataset for pairwise training.')
    if explicit_targets and explicit_targets != [explicit_target]:
      raise ValueError(
        f"target_dataset={explicit_target} conflicts with target_datasets={explicit_targets}. "
        "Use a single pairwise target."
      )
    params.target_datasets = explicit_target
    return explicit_target

  if not explicit_targets:
    return ''

  if len(explicit_targets) != 1:
    raise ValueError(
      f"Strict pairwise protocol expects exactly one target dataset, got {explicit_targets}. "
      "Launch one source->target scenario per run."
    )

  if explicit_targets[0] == params.source_dataset:
    raise ValueError('target dataset must differ from source_dataset for pairwise training.')

  params.target_dataset = explicit_targets[0]
  params.target_datasets = explicit_targets[0]
  return explicit_targets[0]


def save_checkpoint(model, optimizer, epoch, best_acc, params, outfile, lr_scheduler=None):
  raw_model = unwrap_model(model)
  state = {
    'epoch': epoch,
    'state': raw_model.state_dict(),
    'optimizer': optimizer.state_dict() if optimizer is not None else None,
    'best_acc': best_acc,
    'params_meta': _build_meta_from_params(params),
  }
  if lr_scheduler is not None:
    state['lr_scheduler'] = lr_scheduler.state_dict()
  else:
    state['lr_scheduler'] = None
  torch.save(state, outfile)


def _warn_on_meta_mismatch(saved_meta, params):
  if not saved_meta:
    return
  fields = [
    'dataset', 'model', 'method', 'train_n_way', 'test_n_way', 'n_shot', 'name',
  ]
  for field in fields:
    saved_value = saved_meta.get(field)
    current_value = getattr(params, field, None)
    if saved_value is not None and current_value is not None and saved_value != current_value:
      warnings.warn(f'Checkpoint field "{field}"={saved_value} differs from current arg {current_value}')


def train(
  base_loader,
  val_loader,
  model,
  optimizer,
  start_epoch,
  stop_epoch,
  params,
  best_acc=0.0,
  lr_scheduler=None,
  target_unlabeled_loader=None,
):

  if is_main_process() and not os.path.isdir(params.checkpoint_dir):
    os.makedirs(params.checkpoint_dir)

  max_acc = best_acc
  total_it = 0
  train_loss_hist = []
  val_acc_hist = []

  epoch_iter = range(start_epoch, stop_epoch)
  if not getattr(params, "disable_progress_bar", False) and is_main_process():
    epoch_iter = progress_range(epoch_iter, desc=f"Train ({params.source_dataset})")

  for epoch in epoch_iter:
    if hasattr(base_loader, 'batch_sampler') and hasattr(base_loader.batch_sampler, 'set_epoch'):
      base_loader.batch_sampler.set_epoch(epoch)
    model.train()
    total_it, train_loss = model.train_loop(
      epoch,
      base_loader,
      optimizer,
      total_it,
      target_unlabeled_loader=target_unlabeled_loader,
      target_ssl_weight=getattr(params, 'target_ssl_weight', 0.0),
      target_ssl_temperature=getattr(params, 'target_ssl_temperature', 0.5),
      target_ssl_ramp_epochs=getattr(params, 'target_ssl_ramp_epochs', 60),
    )
    model.eval()

    barrier()
    if is_main_process():
      train_loss_hist.append(train_loss)
      if val_loader is not None:
        acc = model.test_loop(val_loader, epoch=epoch)
        val_acc_hist.append(acc)
        if acc > max_acc :
          print(f"Epoch {epoch} | best model! save...")
          max_acc = acc
          outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
          save_checkpoint(model, optimizer, epoch, max_acc, params, outfile, lr_scheduler=lr_scheduler)
        else:
          print(f"Epoch {epoch} | GG! best accuracy {max_acc:.6f}")
      else:
        print(f"Epoch {epoch} | source val disabled under strict pairwise protocol; checkpointing latest model.")

      last_outfile = os.path.join(params.checkpoint_dir, 'last_epoch.tar')
      save_checkpoint(model, optimizer, epoch, max_acc, params, last_outfile, lr_scheduler=lr_scheduler)

      if(epoch == stop_epoch-1):
        if val_loader is None:
          best_outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
          save_checkpoint(model, optimizer, epoch, max_acc, params, best_outfile, lr_scheduler=lr_scheduler)
        outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
        save_checkpoint(model, optimizer, epoch, max_acc, params, outfile, lr_scheduler=lr_scheduler)
    barrier()

  return model, max_acc, train_loss_hist, val_acc_hist


def record_test_result(params):
  """
  Evaluate the model on the SOURCE dataset (Novel split).
  """
  acc_file_path = os.path.join(params.checkpoint_dir, 'acc_source_novel.txt')
  acc_file = open(acc_file_path,'w')
  epoch_id = -1
  dataset_name = params.source_dataset
  print('epoch', epoch_id, f"{dataset_name}:")
  print('epoch', epoch_id, f"{dataset_name}:", file=acc_file)
  name = params.name
  n_shot = params.n_shot
  
  # Use dual writer for test output
  dual_writer = DualWriter(acc_file)
  
  test_bestmodel(
    dual_writer,
    name,
    dataset_name,
    n_shot,
    epoch_id,
    shuffle_query_labels=getattr(params, 'shuffle_query_labels', False),
    data_dir=params.data_dir,
    checkpoint_dir=params.checkpoint_dir,
    model_name=params.model,
    clip_model_name=getattr(params, "clip_model_name", "ViT-B/32"),
    text_guide_epsilon=getattr(params, 'text_guide_epsilon', 0),
    text_guide_gradient=getattr(params, 'text_guide_gradient', 0),
    text_weight_mode=getattr(params, 'text_weight_mode', 'fixed'),
    text_weight_k=getattr(params, 'text_weight_k', 0.2),
    text_weight_fixed=getattr(params, 'text_weight_fixed', 1.0),
    text_confidence_floor=getattr(params, 'text_confidence_floor', 0.35),
    gate_min=getattr(params, 'gate_min', 0.0),
    gate_max=getattr(params, 'gate_max', 1.0),
    epsilon_scale_min=getattr(params, 'epsilon_scale_min', 0.5),
    epsilon_scale_max=getattr(params, 'epsilon_scale_max', 1.5),
    use_prompt_ensemble=bool(getattr(params, 'use_prompt_ensemble', 1)),
    text_calibration_weight=getattr(params, 'text_calibration_weight', 0.0),
    no_gnn=getattr(params, 'no_gnn', False),
    target_ssl_temperature=getattr(params, 'target_ssl_temperature', 0.5),
    target_ssl_confidence_threshold=getattr(params, 'target_ssl_confidence_threshold', 0.80),
    target_ssl_max_entropy=getattr(params, 'target_ssl_max_entropy', 0.75),
    eval_episode_relevance=getattr(params, 'eval_episode_relevance', 0),
    eval_proto_refine=getattr(params, 'eval_proto_refine', 0),
    eval_proto_refine_alpha_1shot=getattr(params, 'eval_proto_refine_alpha_1shot', 8.0),
    eval_proto_refine_alpha_5shot=getattr(params, 'eval_proto_refine_alpha_5shot', 4.0),
    eval_proto_refine_affinity_threshold=getattr(params, 'eval_proto_refine_affinity_threshold', 0.55),
    eval_proto_refine_margin_threshold=getattr(params, 'eval_proto_refine_margin_threshold', 0.05),
    eval_proto_refine_min_selected=getattr(params, 'eval_proto_refine_min_selected', 8),
    eval_proto_refine_min_coverage=getattr(params, 'eval_proto_refine_min_coverage', 0.40),
    eval_proto_refine_affinity_threshold_1shot=getattr(params, 'eval_proto_refine_affinity_threshold_1shot', 0.70),
    eval_proto_refine_affinity_threshold_5shot=getattr(params, 'eval_proto_refine_affinity_threshold_5shot', 0.65),
    eval_proto_refine_margin_threshold_1shot=getattr(params, 'eval_proto_refine_margin_threshold_1shot', 0.12),
    eval_proto_refine_margin_threshold_5shot=getattr(params, 'eval_proto_refine_margin_threshold_5shot', 0.10),
    eval_proto_refine_min_selected_1shot=getattr(params, 'eval_proto_refine_min_selected_1shot', 15),
    eval_proto_refine_min_selected_5shot=getattr(params, 'eval_proto_refine_min_selected_5shot', 12),
    eval_proto_refine_min_coverage_1shot=getattr(params, 'eval_proto_refine_min_coverage_1shot', 0.60),
    eval_proto_refine_min_coverage_5shot=getattr(params, 'eval_proto_refine_min_coverage_5shot', 0.50),
    eval_proto_refine_topk_1shot=getattr(params, 'eval_proto_refine_topk_1shot', 15),
    eval_proto_refine_target_guard=getattr(params, 'eval_proto_refine_target_guard', 1),
    eval_proto_refine_target_affinity_tolerance=getattr(params, 'eval_proto_refine_target_affinity_tolerance', 0.0),
    eval_proto_refine_target_margin_tolerance=getattr(params, 'eval_proto_refine_target_margin_tolerance', 0.0),
    eval_proto_refine_target_affinity_gain_1shot=getattr(params, 'eval_proto_refine_target_affinity_gain_1shot', 0.05),
    eval_proto_refine_target_affinity_gain_5shot=getattr(params, 'eval_proto_refine_target_affinity_gain_5shot', 0.02),
    eval_proto_refine_target_margin_gain_1shot=getattr(params, 'eval_proto_refine_target_margin_gain_1shot', 0.05),
    eval_proto_refine_target_margin_gain_5shot=getattr(params, 'eval_proto_refine_target_margin_gain_5shot', 0.02),
    eval_proto_refine_support_loo_guard=getattr(params, 'eval_proto_refine_support_loo_guard', 1),
    eval_proto_refine_support_loo_tolerance=getattr(params, 'eval_proto_refine_support_loo_tolerance', 0.0),
    eval_proto_refine_support_margin_guard=getattr(params, 'eval_proto_refine_support_margin_guard', 1),
    eval_proto_refine_support_margin_tolerance=getattr(params, 'eval_proto_refine_support_margin_tolerance', 0.0),
    eval_num_workers=getattr(params, 'eval_num_workers', 8),
    feature_batch_size=getattr(params, 'feature_batch_size', 64),
    prefetch_factor=getattr(params, 'prefetch_factor', 2),
    n_query=getattr(params, 'n_query', 15),
    n_episodes_test=getattr(params, 'n_episodes_test', 1000),
    enable_dataparallel=not bool(getattr(params, 'distributed', 0)),
  )

  acc_file.close()
  return


# --- main function ---
if __name__=='__main__':
  params = parse_args('train')
  device = setup_distributed(params)

  #fix seed 
  seed = 0
  rank_seed = seed + getattr(params, 'rank', 0)
  if is_main_process():
    print("set seed = %d" % seed)
  random.seed(rank_seed)
  np.random.seed(rank_seed)
  torch.manual_seed(rank_seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(rank_seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = True

  # Apply best-known defaults per shot setting (auto-choose if user未显式覆盖)
  # 5-shot: epsilon范围更激进 [0.3, 1.7]; 1-shot: 保持稳健 [0.5, 1.5]
  # 注意: 这里强制设置默认值，如果需要实验不同配置，请在命令行显式指定
  # [Step1 注意] use_prompt_ensemble 不在这里覆盖，完全由命令行控制
  # Best semantic-guidance preset keeps CLI text-weight settings intact while
  # still applying shot-aware epsilon defaults when the caller does not override them.
  params.gate_min = 0.0
  params.gate_max = 1.0
  if params.n_shot >= 5:
    params.epsilon_scale_min = 0.3
    params.epsilon_scale_max = 1.7
  else:
    params.epsilon_scale_min = 0.7
    params.epsilon_scale_max = 1.3
  
  # [Step1] 打印关键配置用于调试
  if is_main_process():
    print(
      f"[Config] text_weight_mode={params.text_weight_mode}, "
      f"fixed={params.text_weight_fixed}, floor={getattr(params, 'text_confidence_floor', 0.35)}"
    )
    print(f"[Config] gate_range=[{params.gate_min}, {params.gate_max}]")
    print(f"[Config] epsilon_scale_range=[{params.epsilon_scale_min}, {params.epsilon_scale_max}]")
    print(f"[Config] use_prompt_ensemble={getattr(params, 'use_prompt_ensemble', 1)}")
    print(f"[Config] text_calibration_weight={getattr(params, 'text_calibration_weight', 0.0)}")
    print(
      "[Config] target_ssl="
      f"mode={getattr(params, 'target_ssl_mode', 'legacy_pseudo')}, "
      f"thr={getattr(params, 'target_ssl_confidence_threshold', 0.80)}, "
      f"max_entropy={getattr(params, 'target_ssl_max_entropy', 0.75)}, "
      f"view_thr={getattr(params, 'target_ssl_view_threshold', 0.65)}, "
      f"feat_w={getattr(params, 'target_ssl_feature_weight', 0.25)}, "
      f"proto_w={getattr(params, 'target_ssl_prototype_weight', 0.25)}, "
      f"weight_floor={getattr(params, 'target_ssl_weight_floor', 0.0)}"
    )
    print(
      "[Config] eval_proto="
      f"log={getattr(params, 'eval_episode_relevance', 0)}, "
      f"refine={getattr(params, 'eval_proto_refine', 0)}, "
      f"alpha_1={getattr(params, 'eval_proto_refine_alpha_1shot', 8.0)}, "
      f"alpha_5={getattr(params, 'eval_proto_refine_alpha_5shot', 4.0)}, "
      f"aff_thr={getattr(params, 'eval_proto_refine_affinity_threshold', 0.55)}, "
      f"margin_thr={getattr(params, 'eval_proto_refine_margin_threshold', 0.05)}, "
      f"min_selected={getattr(params, 'eval_proto_refine_min_selected', 8)}, "
      f"min_coverage={getattr(params, 'eval_proto_refine_min_coverage', 0.40)}, "
      f"aff_thr_1={getattr(params, 'eval_proto_refine_affinity_threshold_1shot', 0.70)}, "
      f"aff_thr_5={getattr(params, 'eval_proto_refine_affinity_threshold_5shot', 0.65)}, "
      f"margin_thr_1={getattr(params, 'eval_proto_refine_margin_threshold_1shot', 0.12)}, "
      f"margin_thr_5={getattr(params, 'eval_proto_refine_margin_threshold_5shot', 0.10)}, "
      f"topk_1={getattr(params, 'eval_proto_refine_topk_1shot', 15)}, "
      f"target_guard={getattr(params, 'eval_proto_refine_target_guard', 1)}, "
      f"target_aff_tol={getattr(params, 'eval_proto_refine_target_affinity_tolerance', 0.0)}, "
      f"target_margin_tol={getattr(params, 'eval_proto_refine_target_margin_tolerance', 0.0)}, "
      f"target_aff_gain_1={getattr(params, 'eval_proto_refine_target_affinity_gain_1shot', 0.05)}, "
      f"target_aff_gain_5={getattr(params, 'eval_proto_refine_target_affinity_gain_5shot', 0.02)}, "
      f"target_margin_gain_1={getattr(params, 'eval_proto_refine_target_margin_gain_1shot', 0.05)}, "
      f"target_margin_gain_5={getattr(params, 'eval_proto_refine_target_margin_gain_5shot', 0.02)}, "
      f"loo_guard={getattr(params, 'eval_proto_refine_support_loo_guard', 1)}, "
      f"loo_tol={getattr(params, 'eval_proto_refine_support_loo_tolerance', 0.0)}, "
      f"support_guard={getattr(params, 'eval_proto_refine_support_margin_guard', 1)}, "
      f"guard_tol={getattr(params, 'eval_proto_refine_support_margin_tolerance', 0.0)}"
    )
    print(f"[Config] distributed={bool(getattr(params, 'distributed', 0))}, world_size={getattr(params, 'world_size', 1)}, device={device}")
  
  # Handle use_style_prompt parameter (convert to use_text_guidance)
  if hasattr(params, 'use_style_prompt') and params.use_style_prompt:
    params.use_text_guidance = True
  
  # Ensure source_dataset is set (fallback to dataset if not provided)
  if not hasattr(params, 'source_dataset') or not params.source_dataset:
    params.source_dataset = params.dataset

  pairwise_target = _resolve_pairwise_target(params)
  strict_pairwise_protocol = bool(pairwise_target)
  
  # unify dataset aliases
  params.dataset = params.source_dataset
  params.testset = getattr(params, "testset", params.source_dataset)
  if not hasattr(params, "split"):
    params.split = 'novel'

  dataset_names = [params.dataset]
  if pairwise_target:
    dataset_names.append(pairwise_target)
  missing_paths = validate_dataset_root(params.data_dir, dataset_names)
  if missing_paths:
    raise FileNotFoundError(
      'Dataset root is not portable/complete for the requested experiment. '
      f'Check --data_dir={params.data_dir}. Missing:\n{missing_paths}'
    )

  # output and tensorboard dir
  params.tf_dir = '%s/log/%s'%(params.save_dir, params.name)
  params.checkpoint_dir = '%s/checkpoints/%s'%(params.save_dir, params.name)
  if is_main_process() and not os.path.isdir(params.checkpoint_dir):
    os.makedirs(params.checkpoint_dir)

  # dataloader
  if is_main_process():
    print('\n--- prepare dataloader ---')
    print('  train with single seen domain {}'.format(params.dataset))
    if strict_pairwise_protocol:
      print(f'  strict pairwise target domain {pairwise_target}')
  base_file  = os.path.join(params.data_dir, params.dataset, 'base.json')
  val_file   = os.path.join(params.data_dir, params.dataset, 'val.json')
  novel_file = os.path.join(params.data_dir, params.dataset, 'novel.json')

  def _count_classes(json_path):
    try:
      import json
      with open(json_path, 'r') as f:
        meta = json.load(f)
      return len(set(meta.get('image_labels', [])))
    except Exception:
      return 0

  # model
  if is_main_process():
    print('\n--- build model ---')
  image_size = 224
  
  # Preserve the legacy automatic rule unless a hardware-specific training
  # episode size is explicitly requested. This does not change n_shot or the
  # evaluation-side --n_query.
  legacy_train_n_query = max(1, int(16 * params.test_n_way / params.train_n_way))
  train_n_query_override = getattr(params, 'train_n_query', None)
  n_query = legacy_train_n_query if train_n_query_override is None else int(train_n_query_override)
  if n_query < 1:
    raise ValueError(f'train_n_query must be >= 1, got {n_query}')
  loader_pin_memory = bool(getattr(params, 'pin_memory', 1))
  loader_persistent_workers = bool(getattr(params, 'persistent_workers', 1))
  if is_main_process():
    print(
      f"[Loader] train_workers_per_rank={params.train_num_workers}, "
      f"eval_workers={params.eval_num_workers}, pin_memory={loader_pin_memory}, "
      f"persistent_workers={loader_persistent_workers}, prefetch_factor={params.prefetch_factor}"
    )

  train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot)
  base_datamgr             = SetDataManager(
    image_size,
    n_query=n_query,
    n_eposide=params.train_episodes,
    num_workers=params.train_num_workers,
    pin_memory=loader_pin_memory,
    persistent_workers=loader_persistent_workers,
    prefetch_factor=params.prefetch_factor,
    sampler_seed=seed,
    world_size=getattr(params, 'world_size', 1),
    rank=getattr(params, 'rank', 0),
    data_root=params.data_dir,
    dataset_name=params.dataset,
    **train_few_shot_params,
  )
  if strict_pairwise_protocol:
    source_train_meta, source_train_splits = load_dataset_meta(
      params.dataset,
      splits=("base", "val"),
      data_root=params.data_dir,
    )
    if is_main_process():
      print(f"[Source] strict pairwise training uses all labeled source data from splits: {source_train_splits}")
    base_loader = base_datamgr.get_data_loader(source_train_meta, aug=params.train_aug)
  else:
    base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)

  test_few_shot_params     = dict(n_way = params.test_n_way, n_support = params.n_shot)
  val_loader               = None
  if not strict_pairwise_protocol:
    val_classes = _count_classes(val_file)
    if val_classes < params.test_n_way:
      raise ValueError(
        f'Validation split has only {val_classes} classes but test_n_way={params.test_n_way}. '
        f'Cannot proceed - need at least {params.test_n_way} classes in val.json'
      )
  if is_main_process() and not strict_pairwise_protocol:
    val_datamgr              = SetDataManager(
      image_size,
      n_query=n_query,
      n_eposide=params.val_episodes,
      num_workers=params.eval_num_workers,
      pin_memory=loader_pin_memory,
      persistent_workers=loader_persistent_workers,
      prefetch_factor=params.prefetch_factor,
      sampler_seed=seed,
      data_root=params.data_dir,
      dataset_name=params.dataset,
      **test_few_shot_params,
    )
    val_loader               = val_datamgr.get_data_loader( val_file, aug = False)

  target_unlabeled_loader = None
  if pairwise_target:
    target_unlabeled_loader = build_unlabeled_dataloader(
      pairwise_target,
      image_size=image_size,
      batch_size=getattr(params, 'target_unlabeled_batch_size', 64),
      data_root=params.data_dir,
      split='unlabeled',
      num_workers=params.train_num_workers,
      pin_memory=loader_pin_memory,
      persistent_workers=loader_persistent_workers,
      prefetch_factor=params.prefetch_factor,
    )
    if target_unlabeled_loader is None:
      raise FileNotFoundError(
        f'No unlabeled.json found for target dataset {pairwise_target} under {params.data_dir}. '
        'Run scripts/restructure_datasets.py with --protocol dmpc first.'
      )
    if is_main_process():
      print(
        f"[Target] unlabeled split ready: dataset={pairwise_target}, "
        f"samples={len(target_unlabeled_loader.dataset)}, batch_size={params.target_unlabeled_batch_size}, "
        f"ssl_weight={params.target_ssl_weight}, temp={params.target_ssl_temperature}"
      )

  # Model instantiation with text guidance support
  model = StyleAdvGNN(
    model_dict[params.model],
    tf_path=params.tf_dir,
    device=device,
    dataset_name=params.source_dataset,
    data_root=params.data_dir,
    text_guide_epsilon=params.text_guide_epsilon,
    text_guide_gradient=params.text_guide_gradient,
    clip_model_name=params.clip_model_name,
    # [消融实验] 超参数
    text_weight_mode=getattr(params, 'text_weight_mode', 'adaptive'),
    text_weight_k=getattr(params, 'text_weight_k', 0.2),
    text_weight_fixed=getattr(params, 'text_weight_fixed', 0.5),
    text_confidence_floor=getattr(params, 'text_confidence_floor', 0.35),
    gate_min=getattr(params, 'gate_min', 0.0),
    gate_max=getattr(params, 'gate_max', 1.0),
    epsilon_scale_min=getattr(params, 'epsilon_scale_min', 0.5),
    epsilon_scale_max=getattr(params, 'epsilon_scale_max', 1.5),
    epsilon_block1=getattr(params, 'epsilon_block1', 0.8),
    epsilon_block2=getattr(params, 'epsilon_block2', 0.08),
    epsilon_block3=getattr(params, 'epsilon_block3', 0.008),
    style_attack_mode=getattr(params, 'style_attack_mode', 'progressive'),
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
    target_ssl_temperature=getattr(params, 'target_ssl_temperature', 0.5),
    target_ssl_mode=getattr(params, 'target_ssl_mode', 'legacy_pseudo'),
    target_ssl_confidence_threshold=getattr(params, 'target_ssl_confidence_threshold', 0.80),
    target_ssl_max_entropy=getattr(params, 'target_ssl_max_entropy', 0.75),
    target_ssl_view_threshold=getattr(params, 'target_ssl_view_threshold', 0.65),
    target_ssl_feature_weight=getattr(params, 'target_ssl_feature_weight', 0.25),
    target_ssl_prototype_weight=getattr(params, 'target_ssl_prototype_weight', 0.25),
    target_ssl_weight_floor=getattr(params, 'target_ssl_weight_floor', 0.0),
    eval_episode_relevance=getattr(params, 'eval_episode_relevance', 0),
    eval_proto_refine=getattr(params, 'eval_proto_refine', 0),
    eval_proto_refine_alpha_1shot=getattr(params, 'eval_proto_refine_alpha_1shot', 8.0),
    eval_proto_refine_alpha_5shot=getattr(params, 'eval_proto_refine_alpha_5shot', 4.0),
    eval_proto_refine_affinity_threshold=getattr(params, 'eval_proto_refine_affinity_threshold', 0.55),
    eval_proto_refine_margin_threshold=getattr(params, 'eval_proto_refine_margin_threshold', 0.05),
    eval_proto_refine_min_selected=getattr(params, 'eval_proto_refine_min_selected', 8),
    eval_proto_refine_min_coverage=getattr(params, 'eval_proto_refine_min_coverage', 0.40),
    eval_proto_refine_affinity_threshold_1shot=getattr(params, 'eval_proto_refine_affinity_threshold_1shot', 0.70),
    eval_proto_refine_affinity_threshold_5shot=getattr(params, 'eval_proto_refine_affinity_threshold_5shot', 0.65),
    eval_proto_refine_margin_threshold_1shot=getattr(params, 'eval_proto_refine_margin_threshold_1shot', 0.12),
    eval_proto_refine_margin_threshold_5shot=getattr(params, 'eval_proto_refine_margin_threshold_5shot', 0.10),
    eval_proto_refine_min_selected_1shot=getattr(params, 'eval_proto_refine_min_selected_1shot', 15),
    eval_proto_refine_min_selected_5shot=getattr(params, 'eval_proto_refine_min_selected_5shot', 12),
    eval_proto_refine_min_coverage_1shot=getattr(params, 'eval_proto_refine_min_coverage_1shot', 0.60),
    eval_proto_refine_min_coverage_5shot=getattr(params, 'eval_proto_refine_min_coverage_5shot', 0.50),
    eval_proto_refine_topk_1shot=getattr(params, 'eval_proto_refine_topk_1shot', 15),
    eval_proto_refine_target_guard=getattr(params, 'eval_proto_refine_target_guard', 1),
    eval_proto_refine_target_affinity_tolerance=getattr(params, 'eval_proto_refine_target_affinity_tolerance', 0.0),
    eval_proto_refine_target_margin_tolerance=getattr(params, 'eval_proto_refine_target_margin_tolerance', 0.0),
    eval_proto_refine_target_affinity_gain_1shot=getattr(params, 'eval_proto_refine_target_affinity_gain_1shot', 0.05),
    eval_proto_refine_target_affinity_gain_5shot=getattr(params, 'eval_proto_refine_target_affinity_gain_5shot', 0.02),
    eval_proto_refine_target_margin_gain_1shot=getattr(params, 'eval_proto_refine_target_margin_gain_1shot', 0.05),
    eval_proto_refine_target_margin_gain_5shot=getattr(params, 'eval_proto_refine_target_margin_gain_5shot', 0.02),
    eval_proto_refine_support_loo_guard=getattr(params, 'eval_proto_refine_support_loo_guard', 1),
    eval_proto_refine_support_loo_tolerance=getattr(params, 'eval_proto_refine_support_loo_tolerance', 0.0),
    eval_proto_refine_support_margin_guard=getattr(params, 'eval_proto_refine_support_margin_guard', 1),
    eval_proto_refine_support_margin_tolerance=getattr(params, 'eval_proto_refine_support_margin_tolerance', 0.0),
    **train_few_shot_params,
  )
  model = model.to(device)
  model.distributed = bool(getattr(params, 'distributed', 0))
  model.disable_style_adv_generator = bool(getattr(params, 'disable_style_adv_generator', False))
  if is_main_process() and model.disable_style_adv_generator:
    print('[Config] disable_style_adv_generator=True')

  # load model
  start_epoch = params.start_epoch
  stop_epoch = params.stop_epoch
  optimizer = torch.optim.Adam(model.parameters())
  lr_scheduler = None
  best_acc = 0.0

  if params.resume_from:
    map_location = device
    if is_main_process():
      print(f'  resume from checkpoint: {params.resume_from}')
    checkpoint = torch.load(params.resume_from, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint['state'], strict=False)
    if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
      optimizer.load_state_dict(checkpoint['optimizer'])
    if lr_scheduler is not None and checkpoint.get('lr_scheduler') is not None:
      lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    start_epoch = checkpoint.get('epoch', -1) + 1
    best_acc = checkpoint.get('best_acc', 0.0)
    _warn_on_meta_mismatch(checkpoint.get('params_meta', {}), params)
  elif params.resume != '':
    resume_file = get_resume_file('%s/checkpoints/%s'%(params.save_dir, params.resume), params.resume_epoch)
    if resume_file is not None:
      tmp = torch.load(resume_file, map_location=device, weights_only=False)
      start_epoch = tmp.get('epoch', -1)+1
      model.load_state_dict(tmp['state'], strict=False)
      if 'optimizer' in tmp and tmp['optimizer'] is not None:
        optimizer.load_state_dict(tmp['optimizer'])
      if lr_scheduler is not None and tmp.get('lr_scheduler') is not None:
        lr_scheduler.load_state_dict(tmp['lr_scheduler'])
      best_acc = tmp.get('best_acc', 0.0)
      _warn_on_meta_mismatch(tmp.get('params_meta', {}), params)
      if is_main_process():
        print('  resume the training with at {} epoch (model file {})'.format(start_epoch, params.resume))
  else:
    if params.warmup == 'gg3b0':
      if bool(getattr(params, 'require_warmup', 0)):
        raise FileNotFoundError('require_warmup=1 is incompatible with warmup=gg3b0 (scratch training).')
      if is_main_process():
        print('[info] warmup=gg3b0, train feature encoder from scratch.')
    else:
      warmup_dir = '%s/checkpoints/%s'%(params.save_dir, params.warmup)
      resume_file = get_resume_file(warmup_dir)
      if resume_file is None:
        if bool(getattr(params, 'require_warmup', 0)):
          raise FileNotFoundError(f'Required warmup checkpoint not found under {warmup_dir}')
        if is_main_process():
          print(f'[warn] Warmup checkpoint not found under {warmup_dir}, training from scratch.')
      else:
        state = load_warmup_state(warmup_dir)
        model.feature.load_state_dict(state, strict=False)

  if bool(getattr(params, 'distributed', 0)):
    sync_model_state(model)
  barrier()

  import time
  start = time.perf_counter()
  # training
  if is_main_process():
    print('\n--- start the training ---')
  model, best_acc, train_loss_hist, val_acc_hist = train(
    base_loader,
    val_loader,
    model,
    optimizer,
    start_epoch,
    stop_epoch,
    params,
    best_acc=best_acc,
    lr_scheduler=lr_scheduler,
    target_unlabeled_loader=target_unlabeled_loader,
  )
  end = time.perf_counter()
  if is_main_process():
    if params.stop_epoch > 0:
      runtime_info = f'Running time: {end-start} Seconds: {(end-start)/60} Min: {(end-start)/60/params.stop_epoch} Min per epoch'
      print(runtime_info)
    else:
      runtime_info = f'Running time: {end-start} Seconds: {(end-start)/60} Min'
      print(runtime_info)

  # plotting
  try:
    plot_dir = os.path.join(params.save_dir, "results", "plots", params.name)
    epochs = list(range(start_epoch, stop_epoch))
    if is_main_process() and train_loss_hist:
      plot_curve(
        epochs,
        train_loss_hist,
        xlabel="Epoch",
        ylabel="Train Loss",
        title=f"Train Loss ({params.source_dataset})",
        save_path=os.path.join(plot_dir, "train_loss.png"),
      )
    if is_main_process() and val_acc_hist and len(val_acc_hist) == len(epochs):
      plot_curve(
        epochs,
        val_acc_hist,
        xlabel="Epoch",
        ylabel="Val Acc",
        title=f"Val Acc ({params.source_dataset})",
        save_path=os.path.join(plot_dir, "val_acc.png"),
      )
  except Exception as plot_exc:
    if is_main_process():
      print(f"[warn] plotting failed: {plot_exc}")

  was_distributed = bool(getattr(params, 'distributed', 0))
  main_process = is_main_process()
  if was_distributed:
    barrier()
    cleanup_distributed()
    if not main_process:
      raise SystemExit(0)
    params.distributed = 0

  # --- 1. Optional Test on Source Dataset (Novel Split) ---
  skip_source_test = bool(getattr(params, 'skip_source_test', 0))
  source_novel_classes = _count_classes(novel_file)
  if main_process:
    if skip_source_test:
      print("\n[Testing] Source-domain novel evaluation skipped (skip_source_test=1).")
    elif source_novel_classes < params.test_n_way:
      print(
        f"\n[Testing] Source-domain novel evaluation skipped: "
        f"novel.json has only {source_novel_classes} classes, need at least {params.test_n_way}."
      )
    else:
      print("\n[Testing] Evaluating on Source Domain (Novel Split)...")
      record_test_result(params)
  
  # --- 2. Custom Cross-Domain Testing ---
  # Controlled by --do_bscdfsl flag (1=enabled, 0=disabled)
  do_cross_domain = getattr(params, 'do_bscdfsl', 0)
  
  if main_process and do_cross_domain and hasattr(params, 'target_datasets') and params.target_datasets and params.target_datasets != 'None':
    targets = [t.strip() for t in params.target_datasets.split(',') if t.strip()]
    if targets:
      print(f"\n[Testing] Cross-domain testing enabled (do_bscdfsl={do_cross_domain})")
      print(f"[Testing] Found target datasets: {targets}")
      print("[Testing] Starting Cross-Domain Evaluation...")
      
      for target in targets:
        # Skip if target is the same as source to avoid double testing
        if target == params.dataset:
          continue
          
        print(f"--- Testing on Target Domain: {target} ---")
        acc_file_path = os.path.join(params.checkpoint_dir, f'acc_{target}.txt')
        
        # Open file and call the generic test function
        with open(acc_file_path, 'w') as f:
          print(f'epoch -1 {target}:', file=f)
          dual_writer = DualWriter(f)
          try:
            test_bestmodel(
              dual_writer,
              params.name,
              target, # dataset_name (loads target dataset config)
              params.n_shot,
              -1, # epoch_id
              shuffle_query_labels=getattr(params, 'shuffle_query_labels', False),
              data_dir=params.data_dir,
              checkpoint_dir=params.checkpoint_dir,
              model_name=params.model,
              clip_model_name=getattr(params, "clip_model_name", "ViT-B/32"),
              text_guide_epsilon=getattr(params, 'text_guide_epsilon', 0),
              text_guide_gradient=getattr(params, 'text_guide_gradient', 0),
              text_weight_mode=getattr(params, 'text_weight_mode', 'fixed'),
              text_weight_k=getattr(params, 'text_weight_k', 0.2),
              text_weight_fixed=getattr(params, 'text_weight_fixed', 1.0),
              text_confidence_floor=getattr(params, 'text_confidence_floor', 0.35),
              gate_min=getattr(params, 'gate_min', 0.0),
              gate_max=getattr(params, 'gate_max', 1.0),
              epsilon_scale_min=getattr(params, 'epsilon_scale_min', 0.5),
              epsilon_scale_max=getattr(params, 'epsilon_scale_max', 1.5),
              use_prompt_ensemble=bool(getattr(params, 'use_prompt_ensemble', 1)),
              text_calibration_weight=getattr(params, 'text_calibration_weight', 0.0),
              no_gnn=getattr(params, 'no_gnn', False),
              target_ssl_temperature=getattr(params, 'target_ssl_temperature', 0.5),
              target_ssl_confidence_threshold=getattr(params, 'target_ssl_confidence_threshold', 0.80),
              target_ssl_max_entropy=getattr(params, 'target_ssl_max_entropy', 0.75),
              eval_episode_relevance=getattr(params, 'eval_episode_relevance', 0),
              eval_proto_refine=getattr(params, 'eval_proto_refine', 0),
              eval_proto_refine_alpha_1shot=getattr(params, 'eval_proto_refine_alpha_1shot', 8.0),
              eval_proto_refine_alpha_5shot=getattr(params, 'eval_proto_refine_alpha_5shot', 4.0),
              eval_proto_refine_affinity_threshold=getattr(params, 'eval_proto_refine_affinity_threshold', 0.55),
              eval_proto_refine_margin_threshold=getattr(params, 'eval_proto_refine_margin_threshold', 0.05),
              eval_proto_refine_min_selected=getattr(params, 'eval_proto_refine_min_selected', 8),
              eval_proto_refine_min_coverage=getattr(params, 'eval_proto_refine_min_coverage', 0.40),
              eval_proto_refine_affinity_threshold_1shot=getattr(params, 'eval_proto_refine_affinity_threshold_1shot', 0.70),
              eval_proto_refine_affinity_threshold_5shot=getattr(params, 'eval_proto_refine_affinity_threshold_5shot', 0.65),
              eval_proto_refine_margin_threshold_1shot=getattr(params, 'eval_proto_refine_margin_threshold_1shot', 0.12),
              eval_proto_refine_margin_threshold_5shot=getattr(params, 'eval_proto_refine_margin_threshold_5shot', 0.10),
              eval_proto_refine_min_selected_1shot=getattr(params, 'eval_proto_refine_min_selected_1shot', 15),
              eval_proto_refine_min_selected_5shot=getattr(params, 'eval_proto_refine_min_selected_5shot', 12),
              eval_proto_refine_min_coverage_1shot=getattr(params, 'eval_proto_refine_min_coverage_1shot', 0.60),
              eval_proto_refine_min_coverage_5shot=getattr(params, 'eval_proto_refine_min_coverage_5shot', 0.50),
              eval_proto_refine_topk_1shot=getattr(params, 'eval_proto_refine_topk_1shot', 15),
              eval_proto_refine_target_guard=getattr(params, 'eval_proto_refine_target_guard', 1),
              eval_proto_refine_target_affinity_tolerance=getattr(params, 'eval_proto_refine_target_affinity_tolerance', 0.0),
              eval_proto_refine_target_margin_tolerance=getattr(params, 'eval_proto_refine_target_margin_tolerance', 0.0),
              eval_proto_refine_target_affinity_gain_1shot=getattr(params, 'eval_proto_refine_target_affinity_gain_1shot', 0.05),
              eval_proto_refine_target_affinity_gain_5shot=getattr(params, 'eval_proto_refine_target_affinity_gain_5shot', 0.02),
              eval_proto_refine_target_margin_gain_1shot=getattr(params, 'eval_proto_refine_target_margin_gain_1shot', 0.05),
              eval_proto_refine_target_margin_gain_5shot=getattr(params, 'eval_proto_refine_target_margin_gain_5shot', 0.02),
              eval_proto_refine_support_loo_guard=getattr(params, 'eval_proto_refine_support_loo_guard', 1),
              eval_proto_refine_support_loo_tolerance=getattr(params, 'eval_proto_refine_support_loo_tolerance', 0.0),
              eval_proto_refine_support_margin_guard=getattr(params, 'eval_proto_refine_support_margin_guard', 1),
              eval_proto_refine_support_margin_tolerance=getattr(params, 'eval_proto_refine_support_margin_tolerance', 0.0),
              eval_num_workers=getattr(params, 'eval_num_workers', 8),
              feature_batch_size=getattr(params, 'feature_batch_size', 64),
              prefetch_factor=getattr(params, 'prefetch_factor', 2),
              n_query=getattr(params, 'n_query', 15),
              n_episodes_test=getattr(params, 'n_episodes_test', 1000),
              enable_dataparallel=not bool(getattr(params, 'distributed', 0)),
            )
          except Exception as e:
            print(f"[Error] Failed to test on {target}: {e}")
            
  else:
    if main_process and do_cross_domain:
      print("\n[Testing] Cross-domain testing enabled but no target_datasets specified. Skipping.")
    elif main_process:
      print("\n[Testing] Cross-domain testing disabled (do_bscdfsl=0). Skipping target domain evaluation.")
  if not was_distributed:
    cleanup_distributed()
