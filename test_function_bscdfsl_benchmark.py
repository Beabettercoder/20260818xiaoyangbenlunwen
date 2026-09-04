import os
import h5py
import random
import numpy as np
import torch

import data.feature_loader as feat_loader
from data.datamgr import get_few_shot_datamgr
from methods.backbone_multiblock import model_dict
from methods.StyleAdv_RN_GNN import StyleAdvGNN
from options import get_best_file, get_assigned_file, parse_args
from utils.dataset_utils import parse_target_datasets
from utils.progress_utils import progress_range
from utils.plot_utils import plot_bar
from utils.distributed_utils import is_dist_avail_and_initialized


def build_feature_dict(model, data_loader):
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


def feature_evaluation(cl_data_file, model, n_way=5, n_support=5, n_query=15, shuffle_query_labels=False):
  class_list = cl_data_file.keys()
  select_class = random.sample(list(class_list), n_way)
  z_all = []
  
  # [Step1.1] 收集选中的类名用于文本校准
  selected_class_names = list(select_class)  # HDF5的key就是类名
  
  for cls_idx, cl in enumerate(select_class):
    img_feat = cl_data_file[cl]
    perm_ids = np.random.permutation(len(img_feat)).tolist()
    z_all.append([np.squeeze(img_feat[perm_ids[i]]) for i in range(n_support + n_query)])
  
  z_all = torch.from_numpy(np.array(z_all))
  
  if shuffle_query_labels:
    query_block = z_all[:, n_support:].clone()
    perm = torch.randperm(n_way)
    z_all[:, n_support:] = query_block[perm]

  model.n_query = n_query
  # [Step1.1] 传递 class_names 以启用 Test-time Text Calibration
  scores = model.set_forward(z_all, is_feature=True, class_names=selected_class_names)
  pred = scores.data.cpu().numpy().argmax(axis=1)
  y = np.repeat(range(n_way), n_query)
  acc = np.mean(pred == y) * 100
  return acc


def _evaluate_single_target(params, target_dataset, acc_file, save_epoch):
  n_shot = params.n_shot
  shuffle_query_labels = params.shuffle_query_labels

  shuffle_suffix = " [shuffle query labels]" if shuffle_query_labels else ""
  print(
    f"\nTesting {n_shot}-shot | Source: {params.source_dataset} -> Target: {target_dataset} "
    f"(epoch {save_epoch}){shuffle_suffix}"
  )
  print("Stage 1: extracting features to memory")
  image_size = 224
  split = params.split

  datamgr = get_few_shot_datamgr(
    target_dataset,
    episodic=False,
    image_size=image_size,
    batch_size=getattr(params, "feature_batch_size", 64),
    data_root=params.data_dir,
    split=split,
    num_workers=getattr(params, "eval_num_workers", 8),
    pin_memory=torch.cuda.is_available(),
    persistent_workers=getattr(params, "eval_num_workers", 8) > 0,
    prefetch_factor=getattr(params, "prefetch_factor", 2),
  )
  data_loader = datamgr.get_data_loader(aug=False)

  print("  build feature encoder")
  checkpoint_dir = f"{params.save_dir}/checkpoints/{params.name}"
  if save_epoch != -1:
    modelfile = get_assigned_file(checkpoint_dir, save_epoch)
  else:
    modelfile = get_best_file(checkpoint_dir)
  if modelfile is None or not os.path.isfile(modelfile):
    raise FileNotFoundError(f"未找到特征编码器 checkpoint，请确认 {checkpoint_dir} 下存在 best_model.tar 或指定 epoch 文件")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model_dict[params.model]()
  model = model.to(device)
  tmp = torch.load(modelfile, map_location=device)
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
    torch.cuda.is_available()
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
  iter_num = getattr(params, "n_episodes_test", 1000)
  few_shot_params = dict(n_way=params.test_n_way, n_support=n_shot)
  print("  build metric-based model")
  metric_model = StyleAdvGNN(
    model_dict[params.model],
    dataset_name=target_dataset,
    data_root=params.data_dir,
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
    # [Step1.1: Test-time Text Calibration]
    text_calibration_weight=getattr(params, 'text_calibration_weight', 0.0),
    **few_shot_params,
  )
  metric_model = metric_model.to(device)
  metric_model.eval()

  if save_epoch != -1:
    modelfile = get_assigned_file(checkpoint_dir, save_epoch)
  else:
    modelfile = get_best_file(checkpoint_dir)
  if modelfile is None or not os.path.isfile(modelfile):
    raise FileNotFoundError(f"未找到 metric 模型 checkpoint，请确认 {checkpoint_dir} 下存在 best_model.tar 或指定 epoch 文件")
  tmp = torch.load(modelfile, map_location=device)
  state_to_load = tmp.get("state", tmp.get("model_state"))
  # Drop keys with mismatched shapes (e.g., GNN when n_way changes)
  current = metric_model.state_dict()
  filtered = {}
  dropped = []
  for k, v in state_to_load.items():
    if k not in current or current[k].shape != v.shape:
      dropped.append(k)
      continue
    filtered[k] = v
  # keep silent to avoid noisy state-key dumps
  metric_model.load_state_dict(filtered, strict=False)

  print("  evaluate")
  for _ in progress_range(range(iter_num), desc=f"{target_dataset} eval"):
    acc = feature_evaluation(
      cl_data_file,
      metric_model,
      n_query=getattr(params, "n_query", 15),
      shuffle_query_labels=shuffle_query_labels,
      **few_shot_params,
    )
    acc_all.append(acc)

  acc_all = np.asarray(acc_all)
  acc_mean = np.mean(acc_all)
  acc_std = np.std(acc_all)
  stats_line = "  %d test iterations (%s): Acc = %4.2f%% +- %4.2f%%" % (
    iter_num,
    target_dataset,
    acc_mean,
    1.96 * acc_std / np.sqrt(iter_num),
  )
  print(stats_line + shuffle_suffix)
  print(stats_line + shuffle_suffix, file=acc_file)

  return acc_mean, acc_std


def run_bscdfsl_benchmark(params, acc_file_path=None, save_epoch=None):
  if not hasattr(params, "split"):
    params.split = "novel"
  targets = parse_target_datasets(params.source_dataset, params.target_datasets)
  if not targets:
    print("No cross-domain targets specified; skipping BSCDFSL benchmark.")
    return []

  if save_epoch is None:
    save_epoch = getattr(params, "save_epoch", -1)

  if acc_file_path is None:
    acc_file_path = os.path.join(params.checkpoint_dir, "acc_bscdfsl.txt")
  acc_file = open(acc_file_path, "w")
  print("epoch", save_epoch, f"source={params.source_dataset}", f"targets={','.join(targets)}", file=acc_file)

  result_map = {}
  for target in progress_range(targets, desc=f"BSCDFSL from {params.source_dataset}"):
    acc_mean, acc_std = _evaluate_single_target(params, target, acc_file, save_epoch)
    result_map[target] = {"acc": acc_mean, "std": acc_std}

  acc_file.close()

  # plot bar chart of target accuracies
  try:
    labels = list(result_map.keys())
    values = [result_map[k]["acc"] for k in labels]
    plot_path = os.path.join("results", "plots", params.name, "bscdfsl_acc.png")
    plot_bar(
      labels,
      values,
      xlabel="Target Dataset",
      ylabel="Accuracy (%)",
      title=f"BSCDFSL Acc ({params.source_dataset})",
      save_path=plot_path,
    )
  except Exception as plot_exc:
    print(f"[warn] plotting BSCDFSL failed: {plot_exc}")

  return result_map


if __name__ == "__main__":
  # Parse CLI args using the shared options parser (test mode)
  params = parse_args("test")
  if not hasattr(params, "split"):
    params.split = "novel"
  if not hasattr(params, "checkpoint_dir"):
    params.checkpoint_dir = f"{params.save_dir}/checkpoints/{params.name}"

  run_bscdfsl_benchmark(
    params,
    acc_file_path=os.path.join(params.checkpoint_dir, "acc_bscdfsl.txt"),
    save_epoch=getattr(params, "save_epoch", -1),
  )
