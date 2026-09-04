import numpy as np
import torch
import torch.optim
import os
import random 
import warnings

from methods.backbone_multiblock import model_dict
from data.datamgr import SimpleDataManager, SetDataManager, get_few_shot_datamgr
from methods.StyleAdv_RN_GNN import StyleAdvGNN

from options import parse_args, get_resume_file, load_warmup_state, get_best_file, get_assigned_file
from test_function_bscdfsl_benchmark import run_bscdfsl_benchmark
from utils.progress_utils import progress_range
from utils.plot_utils import plot_curve
from utils.distributed_utils import is_dist_avail_and_initialized
import data.feature_loader as feat_loader
import h5py


def build_feature_dict(model, data_loader):
  """Extract all features directly into memory to avoid HDF5 round trips on shared storage."""
  cl_data_file = {}
  with torch.inference_mode():
    for i, (x, y) in enumerate(data_loader):
      if (i % 10) == 0:
        print("    {:d}/{:d}".format(i, len(data_loader)))
      x = x.to(next(model.parameters()).device, non_blocking=True)
      feats = model(x).data.cpu().numpy()
      labels = y.cpu().numpy()
      for feat, label in zip(feats, labels):
        label = int(label)
        if label not in cl_data_file:
          cl_data_file[label] = []
        cl_data_file[label].append(np.squeeze(feat))
  return cl_data_file


def save_features(model, data_loader, featurefile):
  """Save model features to HDF5 file."""
  f = feat_loader.open_hdf5_file(featurefile, "w")
  max_count = len(data_loader) * data_loader.batch_size
  all_labels = f.create_dataset("all_labels", (max_count,), dtype="i")
  all_feats = None
  count = 0
  with torch.inference_mode():
    for i, (x, y) in enumerate(data_loader):
      if (i % 10) == 0:
        print("    {:d}/{:d}".format(i, len(data_loader)))
      x = x.to(next(model.parameters()).device, non_blocking=True)
      feats = model(x)
      if all_feats is None:
        all_feats = f.create_dataset("all_feats", [max_count] + list(feats.size()[1:]), dtype="f")
      all_feats[count : count + feats.size(0)] = feats.data.cpu().numpy()
      all_labels[count : count + feats.size(0)] = y.cpu().numpy()
      count = count + feats.size(0)

  count_var = f.create_dataset("count", (1,), dtype="i")
  count_var[0] = count
  f.close()


def feature_evaluation(cl_data_file, model, n_way=5, n_support=5, n_query=15, shuffle_query_labels=False, no_gnn=False, return_stats=False):
  """Evaluate model on few-shot tasks using pre-extracted features."""
  class_list = cl_data_file.keys()
  select_class = random.sample(list(class_list), n_way)
  z_all = []
  for cl in select_class:
    img_feat = cl_data_file[cl]
    perm_ids = np.random.permutation(len(img_feat)).tolist()
    z_all.append([np.squeeze(img_feat[perm_ids[i]]) for i in range(n_support + n_query)])
  z_all = torch.from_numpy(np.array(z_all))
  if shuffle_query_labels:
    query_block = z_all[:, n_support:].clone()
    perm = torch.randperm(n_way)
    z_all[:, n_support:] = query_block[perm]

  model.n_query = n_query
  class_ids = [int(cl) for cl in select_class]
  class_names = None
  if hasattr(model, "_get_class_names_for_ids"):
    class_names = model._get_class_names_for_ids(class_ids)
  # Text calibration expects semantic class names, not numeric label ids.
  scores = model.set_forward(z_all, is_feature=True, class_names=class_names, use_gnn=(not no_gnn))
  pred = scores.data.cpu().numpy().argmax(axis=1)
  y = np.repeat(range(n_way), n_query)
  acc = np.mean(pred == y) * 100
  if return_stats:
    return acc, getattr(model, "last_eval_episode_stats", {})
  return acc


def _create_eval_stats_accumulator():
  return {}


def _accumulate_eval_stats(accumulator, episode_stats):
  if not episode_stats:
    return
  for key, value in episode_stats.items():
    if value is None:
      continue
    value = float(value)
    if np.isnan(value) or np.isinf(value):
      continue
    total, count = accumulator.get(key, (0.0, 0))
    accumulator[key] = (total + value, count + 1)


def _finalize_eval_stats(accumulator):
  summary = {}
  for key, (total, count) in accumulator.items():
    if count > 0:
      summary[key] = total / float(count)
  return summary


def _build_meta_from_params(params):
  return dict(
    dataset=params.dataset,
    model=params.model,
    method=params.method,
    train_n_way=params.train_n_way,
    test_n_way=params.test_n_way,
    n_shot=params.n_shot,
    name=params.name,
    use_style_prompt=getattr(params, "use_style_prompt", 0),
    clip_align_weight=getattr(params, "clip_align_weight", 0.0),
    style_prompt_dim=getattr(params, "style_prompt_dim", None),
    clip_model_name=getattr(params, "clip_model_name", None),
  )


def save_checkpoint(model, optimizer, epoch, best_acc, params, outfile, lr_scheduler=None):
  state = {
    'epoch': epoch,
    'state': model.state_dict(),
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
    'use_style_prompt', 'clip_align_weight', 'style_prompt_dim', 'clip_model_name',
  ]
  for field in fields:
    saved_value = saved_meta.get(field)
    current_value = getattr(params, field, None)
    if saved_value is not None and current_value is not None and saved_value != current_value:
      warnings.warn(f'Checkpoint field "{field}"={saved_value} differs from current arg {current_value}')


def train(base_loader, val_loader, model, optimizer, start_epoch, stop_epoch, params, best_acc=0.0, lr_scheduler=None):

  if not os.path.isdir(params.checkpoint_dir):
    os.makedirs(params.checkpoint_dir)

  max_acc = best_acc
  total_it = 0
  train_loss_hist = []
  val_acc_hist = []

  epoch_iter = range(start_epoch, stop_epoch)
  if not getattr(params, "disable_progress_bar", False):
    epoch_iter = progress_range(epoch_iter, desc=f"Train ({params.source_dataset})")

  for epoch in epoch_iter:
    model.train()
    total_it, train_loss = model.train_loop(epoch, base_loader, optimizer, total_it) #model are called by reference, no need to return
    model.eval()

    train_loss_hist.append(train_loss)
    acc = model.test_loop( val_loader)
    val_acc_hist.append(acc)
    if acc > max_acc :
      print("best model! save...")
      max_acc = acc
      outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
      save_checkpoint(model, optimizer, epoch, max_acc, params, outfile, lr_scheduler=lr_scheduler)
    else:
      print("GG! best accuracy {:f}".format(max_acc))

    last_outfile = os.path.join(params.checkpoint_dir, 'last_epoch.tar')
    save_checkpoint(model, optimizer, epoch, max_acc, params, last_outfile, lr_scheduler=lr_scheduler)

    #if ((epoch + 1) % params.save_freq==0) or (epoch==stop_epoch-1):
    if(epoch == stop_epoch-1):
      outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
      save_checkpoint(model, optimizer, epoch, max_acc, params, outfile, lr_scheduler=lr_scheduler)

  return model, max_acc, train_loss_hist, val_acc_hist


def record_test_result(params):
  acc_file_path = os.path.join(params.checkpoint_dir, 'acc_bywzj20251118.txt')
  acc_file = open(acc_file_path,'w')
  epoch_id = -1
  dataset_name = params.source_dataset
  print('epoch', epoch_id, f"{dataset_name}:", file=acc_file)
  name = params.name
  n_shot = params.n_shot
  test_bestmodel(
    acc_file,
    name,
    dataset_name,
    n_shot,
    epoch_id,
    shuffle_query_labels=params.shuffle_query_labels,
    data_dir=params.data_dir,
    checkpoint_dir=params.checkpoint_dir,
    clip_model_name=getattr(params, "clip_model_name", "ViT-B/32"),
    model_name=getattr(params, "model", "ResNet10"),
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
  )

  acc_file.close()
  return


def record_test_result_bscdfsl(params):
  print('Running BSCDFSL benchmark...')
  run_bscdfsl_benchmark(
    params,
    acc_file_path=os.path.join(params.checkpoint_dir, 'acc_bscdfsl.txt'),
    save_epoch=-1,
  )


# --- main function ---
if __name__=='__main__':
  #fix seed 
  seed = 0
  print("set seed = %d" % seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False 

  # parser argument
  params = parse_args('train')
  # unify dataset aliases
  params.dataset = params.source_dataset
  params.testset = getattr(params, "testset", params.source_dataset)
  if not hasattr(params, "split"):
    params.split = 'novel'

  # output and tensorboard dir
  params.tf_dir = '%s/log/%s'%(params.save_dir, params.name)
  params.checkpoint_dir = '%s/checkpoints/%s'%(params.save_dir, params.name)
  if not os.path.isdir(params.checkpoint_dir):
    os.makedirs(params.checkpoint_dir)

  # dataloader
  print('\n--- prepare dataloader ---')
  print('  train with single seen domain {}'.format(params.dataset))
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

  val_classes = _count_classes(val_file)
  # [FIXED] Do NOT use novel split for validation - that's data leakage!
  # The model should be validated on val.json only, never on novel.json
  if val_classes < params.test_n_way:
    raise ValueError(
      f'Validation split has only {val_classes} classes but test_n_way={params.test_n_way}. '
      f'Cannot proceed - need at least {params.test_n_way} classes in val.json'
    )

  # model
  print('\n--- build model ---')
  image_size = 224
  
  #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
  n_query = max(1, int(16* params.test_n_way/params.train_n_way))

  train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot)
  base_datamgr            = SetDataManager(image_size, n_query = n_query, n_eposide=params.train_episodes, **train_few_shot_params)
  base_loader             = base_datamgr.get_data_loader( base_file , aug = params.train_aug )

  test_few_shot_params     = dict(n_way = params.test_n_way, n_support = params.n_shot)
  val_datamgr             = SetDataManager(image_size, n_query = n_query, n_eposide=params.val_episodes, **test_few_shot_params)
  val_loader              = val_datamgr.get_data_loader( val_file, aug = False)

  model           = StyleAdvGNN(
    model_dict[params.model],
    tf_path=params.tf_dir,
    **train_few_shot_params,
    dataset_name=params.dataset,
    data_root=params.data_dir,
    use_style_prompt=getattr(params, "use_style_prompt", 0),
    clip_align_weight=getattr(params, "clip_align_weight", 0.0),
    style_prompt_dim=getattr(params, "style_prompt_dim", None),
    clip_model_name=getattr(params, "clip_model_name", "ViT-B/32"),
    # [消融实验] 传递所有超参数以保持训练-测试一致性
    text_weight_mode=getattr(params, 'text_weight_mode', 'fixed'),
    text_weight_fixed=getattr(params, 'text_weight_fixed', 1.0),
    gate_min=getattr(params, 'gate_min', 0.0),
    gate_max=getattr(params, 'gate_max', 1.0),
    epsilon_scale_min=getattr(params, 'epsilon_scale_min', 0.5),
    epsilon_scale_max=getattr(params, 'epsilon_scale_max', 1.5),
    # [Step1: PromptEnsemble] 多模板集成
    use_prompt_ensemble=bool(getattr(params, 'use_prompt_ensemble', 1)),
  )
  model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

  # load model
  start_epoch = params.start_epoch
  stop_epoch = params.stop_epoch
  optimizer = torch.optim.Adam(model.parameters())
  lr_scheduler = None
  best_acc = 0.0

  if params.resume_from:
    map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
      tmp = torch.load(resume_file, weights_only=False)
      start_epoch = tmp.get('epoch', -1)+1
      model.load_state_dict(tmp['state'], strict=False)
      if 'optimizer' in tmp and tmp['optimizer'] is not None:
        optimizer.load_state_dict(tmp['optimizer'])
      if lr_scheduler is not None and tmp.get('lr_scheduler') is not None:
        lr_scheduler.load_state_dict(tmp['lr_scheduler'])
      best_acc = tmp.get('best_acc', 0.0)
      _warn_on_meta_mismatch(tmp.get('params_meta', {}), params)
      print('  resume the training with at {} epoch (model file {})'.format(start_epoch, params.resume))
  else:
    if params.warmup == 'gg3b0':
      raise Exception('Must provide the pre-trained feature encoder file using --warmup option!')
    warmup_dir = '%s/checkpoints/%s'%(params.save_dir, params.warmup)
    resume_file = get_resume_file(warmup_dir)
    if resume_file is None:
      print(f'[warn] Warmup checkpoint not found under {warmup_dir}, training from scratch.')
    else:
      state = load_warmup_state(warmup_dir)
      model.feature.load_state_dict(state, strict=False)

  import time
  start = time.perf_counter()
  # training
  print('\n--- start the training ---')
  model, best_acc, train_loss_hist, val_acc_hist = train(base_loader, val_loader, model, optimizer, start_epoch, stop_epoch, params, best_acc=best_acc, lr_scheduler=lr_scheduler)
  end = time.perf_counter()
  if params.stop_epoch > 0:
    print('Running time: %s Seconds: %s Min: %s Min per epoch'%(end-start, (end-start)/60, (end-start)/60/params.stop_epoch))
  else:
    print('Running time: %s Seconds: %s Min'%(end-start, (end-start)/60))

  # plotting
  try:
    plot_dir = os.path.join("results", "plots", params.name)
    epochs = list(range(start_epoch, stop_epoch))
    if train_loss_hist:
      plot_curve(
        epochs,
        train_loss_hist,
        xlabel="Epoch",
        ylabel="Train Loss",
        title=f"Train Loss ({params.source_dataset})",
        save_path=os.path.join(plot_dir, "train_loss.png"),
      )
    if val_acc_hist and len(val_acc_hist) == len(epochs):
      plot_curve(
        epochs,
        val_acc_hist,
        xlabel="Epoch",
        ylabel="Val Acc",
        title=f"Val Acc ({params.source_dataset})",
        save_path=os.path.join(plot_dir, "val_acc.png"),
      )
  except Exception as plot_exc:
    print(f"[warn] plotting failed: {plot_exc}")

  # testing
  record_test_result(params)
  # testing bscdfsl
  if getattr(params, "do_bscdfsl", 0):
    record_test_result_bscdfsl(params)
  else:
    print("Skip BSCDFSL benchmark (do_bscdfsl=0)")


def test_bestmodel(
  acc_file,
  name,
  dataset_name,
  n_shot,
  epoch_id,
  shuffle_query_labels=False,
  data_dir='./data',
  checkpoint_dir=None,
  clip_model_name="ViT-B/32",
  model_name: str = "ResNet10",
  text_guide_epsilon=0,
  text_guide_gradient=0,
  text_weight_mode='fixed',
  text_weight_k=0.2,
  text_weight_fixed=1.0,
  text_confidence_floor=0.35,
  gate_min=0.0,
  gate_max=1.0,
  epsilon_scale_min=0.5,
  epsilon_scale_max=1.5,
  use_prompt_ensemble=True,
  text_calibration_weight=0.0,
  no_gnn=False,
  target_ssl_temperature=0.5,
  target_ssl_confidence_threshold=0.80,
  target_ssl_max_entropy=0.75,
  eval_episode_relevance=0,
  eval_proto_refine=0,
  eval_proto_refine_alpha_1shot=8.0,
  eval_proto_refine_alpha_5shot=4.0,
  eval_proto_refine_affinity_threshold=0.55,
  eval_proto_refine_margin_threshold=0.05,
  eval_proto_refine_min_selected=8,
  eval_proto_refine_min_coverage=0.40,
  eval_proto_refine_affinity_threshold_1shot=0.70,
  eval_proto_refine_affinity_threshold_5shot=0.65,
  eval_proto_refine_margin_threshold_1shot=0.12,
  eval_proto_refine_margin_threshold_5shot=0.10,
  eval_proto_refine_min_selected_1shot=15,
  eval_proto_refine_min_selected_5shot=12,
  eval_proto_refine_min_coverage_1shot=0.60,
  eval_proto_refine_min_coverage_5shot=0.50,
  eval_proto_refine_topk_1shot=15,
  eval_proto_refine_target_guard=1,
  eval_proto_refine_target_affinity_tolerance=0.0,
  eval_proto_refine_target_margin_tolerance=0.0,
  eval_proto_refine_target_affinity_gain_1shot=0.05,
  eval_proto_refine_target_affinity_gain_5shot=0.02,
  eval_proto_refine_target_margin_gain_1shot=0.05,
  eval_proto_refine_target_margin_gain_5shot=0.02,
  eval_proto_refine_support_loo_guard=1,
  eval_proto_refine_support_loo_tolerance=0.0,
  eval_proto_refine_support_margin_guard=1,
  eval_proto_refine_support_margin_tolerance=0.0,
  eval_num_workers=8,
  feature_batch_size=64,
  prefetch_factor=2,
  n_query=15,
  n_episodes_test=1000,
  enable_dataparallel=True,
):
  """
  Test the best model checkpoint on a given dataset.
  
  Args:
    acc_file: File handle to write results to (or DualWriter object)
    name: Experiment name (checkpoint directory)
    dataset_name: Dataset to test on
    n_shot: Number of support samples
    epoch_id: Epoch to load (-1 for best_model)
    shuffle_query_labels: Whether to shuffle query labels
    data_dir: Data directory
    use_style_prompt: Whether to use style prompt
    clip_align_weight: CLIP alignment weight
    style_prompt_dim: Style prompt dimension
    clip_model_name: CLIP model name
  """
  print(f"[test_bestmodel] Testing {name} on {dataset_name} ({n_shot}-shot)")
  
  shuffle_suffix = " [shuffle query labels]" if shuffle_query_labels else ""
  print("Stage 1: extracting features to memory")
  image_size = 224
  split = 'novel'
  
  datamgr = get_few_shot_datamgr(
    dataset_name,
    episodic=False,
    image_size=image_size,
    batch_size=feature_batch_size,
    data_root=data_dir,
    split=split,
    num_workers=eval_num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=eval_num_workers > 0,
    prefetch_factor=prefetch_factor,
  )
  data_loader = datamgr.get_data_loader(aug=False)
  
  print("  build feature encoder")
  if checkpoint_dir is None:
    checkpoint_dir = os.path.join("output", "checkpoints", name)
  if epoch_id != -1:
    modelfile = get_assigned_file(checkpoint_dir, epoch_id)
  else:
    modelfile = get_best_file(checkpoint_dir)
  if modelfile is None or not os.path.isfile(modelfile):
    raise FileNotFoundError(f"未找到特征编码器 checkpoint，请确认 {checkpoint_dir} 下存在 best_model.tar 或指定 epoch 文件")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  if model_name not in model_dict:
    raise KeyError(f"model '{model_name}' not in model_dict. Available: {list(model_dict.keys())}")
  model = model_dict[model_name]()
  model = model.to(device)
  tmp = torch.load(modelfile, map_location=device, weights_only=False)
  state = tmp.get("state", tmp.get("model_state"))
  if state is None:
    raise KeyError("No state in checkpoint")
  for key in list(state.keys()):
    if "feature." in key and "gamma" not in key and "beta" not in key:
      newkey = key.replace("feature.", "")
      state[newkey] = state.pop(key)
    else:
      state.pop(key)
  model.load_state_dict(state, strict=False)
  use_dataparallel = (
    enable_dataparallel
    and torch.cuda.is_available()
    and torch.cuda.device_count() > 1
    and not is_dist_avail_and_initialized()
  )
  if use_dataparallel:
    print(f"  feature encoder DataParallel enabled on {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)
  model.eval()
  
  print("  extract features...")
  cl_data_file = build_feature_dict(model, data_loader)
  
  print("Stage 2: evaluate")
  acc_all = []
  eval_stats_acc = _create_eval_stats_accumulator()
  iter_num = n_episodes_test
  few_shot_params = dict(n_way=5, n_support=n_shot)
  print("  build metric-based model")
  metric_model = StyleAdvGNN(
    model_dict[model_name],
    dataset_name=dataset_name,
    data_root=data_dir,
    text_guide_epsilon=text_guide_epsilon,
    text_guide_gradient=text_guide_gradient,
    clip_model_name=clip_model_name,
    # [消融实验] 传递所有超参数以保持训练-测试一致性
    text_weight_mode=text_weight_mode,
    text_weight_k=text_weight_k,
    text_weight_fixed=text_weight_fixed,
    text_confidence_floor=text_confidence_floor,
    gate_min=gate_min,
    gate_max=gate_max,
    epsilon_scale_min=epsilon_scale_min,
    epsilon_scale_max=epsilon_scale_max,
    # [Step1: PromptEnsemble] 多模板集成
    use_prompt_ensemble=bool(use_prompt_ensemble),
    # [Step1.1: Test-time Text Calibration]
    text_calibration_weight=text_calibration_weight,
    target_ssl_temperature=target_ssl_temperature,
    target_ssl_confidence_threshold=target_ssl_confidence_threshold,
    target_ssl_max_entropy=target_ssl_max_entropy,
    eval_episode_relevance=eval_episode_relevance,
    eval_proto_refine=eval_proto_refine,
    eval_proto_refine_alpha_1shot=eval_proto_refine_alpha_1shot,
    eval_proto_refine_alpha_5shot=eval_proto_refine_alpha_5shot,
    eval_proto_refine_affinity_threshold=eval_proto_refine_affinity_threshold,
    eval_proto_refine_margin_threshold=eval_proto_refine_margin_threshold,
    eval_proto_refine_min_selected=eval_proto_refine_min_selected,
    eval_proto_refine_min_coverage=eval_proto_refine_min_coverage,
    eval_proto_refine_affinity_threshold_1shot=eval_proto_refine_affinity_threshold_1shot,
    eval_proto_refine_affinity_threshold_5shot=eval_proto_refine_affinity_threshold_5shot,
    eval_proto_refine_margin_threshold_1shot=eval_proto_refine_margin_threshold_1shot,
    eval_proto_refine_margin_threshold_5shot=eval_proto_refine_margin_threshold_5shot,
    eval_proto_refine_min_selected_1shot=eval_proto_refine_min_selected_1shot,
    eval_proto_refine_min_selected_5shot=eval_proto_refine_min_selected_5shot,
    eval_proto_refine_min_coverage_1shot=eval_proto_refine_min_coverage_1shot,
    eval_proto_refine_min_coverage_5shot=eval_proto_refine_min_coverage_5shot,
    eval_proto_refine_topk_1shot=eval_proto_refine_topk_1shot,
    eval_proto_refine_target_guard=eval_proto_refine_target_guard,
    eval_proto_refine_target_affinity_tolerance=eval_proto_refine_target_affinity_tolerance,
    eval_proto_refine_target_margin_tolerance=eval_proto_refine_target_margin_tolerance,
    eval_proto_refine_target_affinity_gain_1shot=eval_proto_refine_target_affinity_gain_1shot,
    eval_proto_refine_target_affinity_gain_5shot=eval_proto_refine_target_affinity_gain_5shot,
    eval_proto_refine_target_margin_gain_1shot=eval_proto_refine_target_margin_gain_1shot,
    eval_proto_refine_target_margin_gain_5shot=eval_proto_refine_target_margin_gain_5shot,
    eval_proto_refine_support_loo_guard=eval_proto_refine_support_loo_guard,
    eval_proto_refine_support_loo_tolerance=eval_proto_refine_support_loo_tolerance,
    eval_proto_refine_support_margin_guard=eval_proto_refine_support_margin_guard,
    eval_proto_refine_support_margin_tolerance=eval_proto_refine_support_margin_tolerance,
    **few_shot_params,
  )
  metric_model = metric_model.to(device)
  metric_model.eval()
  
  if epoch_id != -1:
    modelfile = get_assigned_file(checkpoint_dir, epoch_id)
  else:
    modelfile = get_best_file(checkpoint_dir)
  if modelfile is None or not os.path.isfile(modelfile):
    raise FileNotFoundError(f"未找到 metric 模型 checkpoint，请确认 {checkpoint_dir} 下存在 best_model.tar 或指定 epoch 文件")
  tmp = torch.load(modelfile, map_location=device, weights_only=False)
  state_to_load = tmp.get("state", tmp.get("model_state"))
  # Drop keys with mismatched shapes
  current = metric_model.state_dict()
  filtered = {}
  dropped = []
  for k, v in state_to_load.items():
    if k not in current or current[k].shape != v.shape:
      dropped.append(k)
      continue
    filtered[k] = v
  metric_model.load_state_dict(filtered, strict=False)
  
  print("  evaluate")
  for _ in progress_range(range(iter_num), desc=f"{dataset_name} eval"):
    acc, episode_stats = feature_evaluation(
      cl_data_file,
      metric_model,
      n_query=n_query,
      shuffle_query_labels=shuffle_query_labels,
      no_gnn=no_gnn,
      return_stats=True,
      **few_shot_params,
    )
    acc_all.append(acc)
    _accumulate_eval_stats(eval_stats_acc, episode_stats)
  
  acc_all = np.asarray(acc_all)
  acc_mean = np.mean(acc_all)
  acc_std = np.std(acc_all)
  eval_stats_summary = _finalize_eval_stats(eval_stats_acc)
  stats_line = "  %d test iterations (%s): Acc = %4.2f%% +- %4.2f%%" % (
    iter_num,
    dataset_name,
    acc_mean,
    1.96 * acc_std / np.sqrt(iter_num),
  )
  
  # Write to file (supports both file objects and DualWriter)
  output_line = stats_line + shuffle_suffix
  print(output_line)
  if hasattr(acc_file, 'write'):
    acc_file.write(output_line + '\n')
  else:
    print(output_line, file=acc_file)

  if eval_stats_summary:
    summary_lines = [
      "  Episode relevance (all_selected): affinity = {all_selected_prototype_affinity:.4f}, margin = {all_selected_prototype_margin:.4f}, coverage = {all_selected_class_coverage:.4f}, count = {all_selected_count:.2f}, ratio = {all_selected_ratio:.4f}".format(**eval_stats_summary),
      "  Episode relevance (gate_passed): affinity = {gate_passed_prototype_affinity:.4f}, margin = {gate_passed_prototype_margin:.4f}, coverage = {gate_passed_class_coverage:.4f}, count = {gate_passed_count:.2f}, ratio = {gate_passed_ratio:.4f}, refine_candidate = {refinement_candidate:.4f}, refine_applied = {refinement_applied:.4f}".format(**eval_stats_summary),
    ]
    if 'support_loo_before' in eval_stats_summary and 'support_loo_after' in eval_stats_summary:
      summary_lines.append(
        "  Support LOO: before = {before:.4f}, candidate = {candidate:.4f}, after = {after:.4f}".format(
          before=eval_stats_summary.get('support_loo_before', float('nan')),
          candidate=eval_stats_summary.get('support_loo_candidate', float('nan')),
          after=eval_stats_summary.get('support_loo_after', float('nan')),
        )
      )
    if 'support_margin_before' in eval_stats_summary and 'support_margin_after' in eval_stats_summary:
      summary_lines.append(
        "  Support margin: before = {before:.4f}, candidate = {candidate:.4f}, after = {after:.4f}, guard_passed = {guard:.4f}".format(
          before=eval_stats_summary.get('support_margin_before', float('nan')),
          candidate=eval_stats_summary.get('support_margin_candidate', float('nan')),
          after=eval_stats_summary.get('support_margin_after', float('nan')),
          guard=eval_stats_summary.get('refinement_guard_passed', float('nan')),
        )
      )
    if 'target_prototype_affinity_before' in eval_stats_summary and 'target_prototype_margin_after' in eval_stats_summary:
      summary_lines.append(
        "  Target proxy: affinity before = {aff_before:.4f}, candidate = {aff_candidate:.4f}, after = {aff_after:.4f}; margin before = {margin_before:.4f}, candidate = {margin_candidate:.4f}, after = {margin_after:.4f}".format(
          aff_before=eval_stats_summary.get('target_prototype_affinity_before', float('nan')),
          aff_candidate=eval_stats_summary.get('target_prototype_affinity_candidate', float('nan')),
          aff_after=eval_stats_summary.get('target_prototype_affinity_after', float('nan')),
          margin_before=eval_stats_summary.get('target_prototype_margin_before', float('nan')),
          margin_candidate=eval_stats_summary.get('target_prototype_margin_candidate', float('nan')),
          margin_after=eval_stats_summary.get('target_prototype_margin_after', float('nan')),
        )
      )
    if 'refinement_guard_passed' in eval_stats_summary:
      summary_lines.append(
        "  Accept guard: target = {target:.4f}, loo = {loo:.4f}, margin = {margin:.4f}, overall = {overall:.4f}, topk_1shot = {topk:.2f}".format(
          target=eval_stats_summary.get('refinement_target_guard_passed', float('nan')),
          loo=eval_stats_summary.get('refinement_loo_guard_passed', float('nan')),
          margin=eval_stats_summary.get('refinement_margin_guard_passed', float('nan')),
          overall=eval_stats_summary.get('refinement_guard_passed', float('nan')),
          topk=eval_stats_summary.get('refinement_topk_cap', float('nan')),
        )
      )
    for line in summary_lines:
      print(line)
      if hasattr(acc_file, 'write'):
        acc_file.write(line + '\n')
      else:
        print(line, file=acc_file)
  
  return acc_mean, acc_std
