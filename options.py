import argparse
import glob
import os

import numpy as np
import torch
from utils.distributed_utils import is_main_process
from utils.path_utils import absolute_path, repository_root


REPO_ROOT = repository_root()


def _add_common_args(parser):
  parser.add_argument('--dataset', default='multi', help='Dataset name or multi for legacy multi-domain mode')
  parser.add_argument('--source_dataset', default='NWPU', help='Source domain for training')
  parser.add_argument('--testset', default='cub', help='Legacy target dataset name')
  parser.add_argument('--model', default='ResNet10', help='Model backbone: Conv{4|6} / ResNet{10|18|34}')
  parser.add_argument('--method', default='baseline', help='Method name')
  parser.add_argument('--train_n_way', default=5, type=int, help='Class count per training episode')
  parser.add_argument('--test_n_way', default=5, type=int, help='Class count per validation/test episode')
  parser.add_argument('--n_shot', default=5, type=int, help='Support samples per class')
  parser.add_argument('--train_aug', action='store_true', help='Enable data augmentation during training')
  parser.add_argument('--name', default='tmp', type=str, help='Experiment name')
  parser.add_argument(
    '--save_dir',
    default=str(REPO_ROOT / 'output'),
    type=str,
    help='Output root directory (defaults to output/ inside this repository)',
  )
  parser.add_argument(
    '--data_dir',
    required=True,
    type=str,
    help='Dataset root directory, containing one folder per dataset',
  )

  parser.add_argument('--use_text_guidance', action='store_true', help='Enable text-guided style attack')
  parser.add_argument('--use_style_prompt', type=int, default=0, help='Alias for enabling text guidance')
  parser.add_argument('--text_guide_epsilon', type=int, default=0, help='Enable text-guided epsilon scaling')
  parser.add_argument('--text_guide_gradient', type=int, default=0, help='Enable text-guided gradient gating')
  parser.add_argument('--clip_model_name', type=str, default='ViT-B/32', help='CLIP model name for text encoding')
  parser.add_argument('--text_guidance_path', default='', type=str, help='Optional path to precomputed text guidance data')
  parser.add_argument('--text_weight_mode', type=str, default='fixed', choices=['adaptive', 'fixed', 'agreement'], help='Text weighting strategy')
  parser.add_argument('--text_weight_k', type=float, default=0.2, help='Adaptive text weight coefficient')
  parser.add_argument('--text_weight_fixed', type=float, default=1.0, help='Fixed text weight value')
  parser.add_argument('--text_confidence_floor', type=float, default=0.35, help='Lower bound for agreement-aware text weighting')
  parser.add_argument('--gate_min', type=float, default=0.0, help='Lower bound for gradient gating')
  parser.add_argument('--gate_max', type=float, default=1.0, help='Upper bound for gradient gating')
  parser.add_argument('--epsilon_scale_min', type=float, default=0.5, help='Lower bound for epsilon scaling')
  parser.add_argument('--epsilon_scale_max', type=float, default=1.5, help='Upper bound for epsilon scaling')
  parser.add_argument('--epsilon_block1', type=float, default=0.8, help='Base epsilon for block1 attack')
  parser.add_argument('--epsilon_block2', type=float, default=0.08, help='Base epsilon for block2 attack')
  parser.add_argument('--epsilon_block3', type=float, default=0.008, help='Base epsilon for block3 attack')
  parser.add_argument('--style_attack_mode', type=str, default='progressive', choices=['progressive', 'independent'], help='Style attack composition mode')
  parser.add_argument('--attack_mode', dest='style_attack_mode', choices=['progressive', 'independent'], help='Legacy alias for --style_attack_mode')
  parser.add_argument('--use_prompt_ensemble', type=int, default=1, help='Use prompt ensemble for text encoding')
  parser.add_argument('--text_calibration_weight', type=float, default=0.0, help='Embedding-level text calibration weight')
  parser.add_argument('--semantic_anchor', type=int, default=0, help='Enable the visual-to-text semantic anchor')
  parser.add_argument('--semantic_anchor_weight', type=float, default=1.0, help='Outer clean-episode weight for the semantic anchor loss')
  parser.add_argument('--semantic_anchor_temperature', type=float, default=0.07, help='Temperature for the visual-to-text semantic anchor')
  parser.add_argument('--semantic_drift_control', type=int, default=0, help='Enable semantic-gradient risk budgets and drift feedback')
  parser.add_argument('--semantic_risk_beta', type=float, default=1.0, help='Semantic-risk penalty used for channel budget allocation')
  parser.add_argument('--semantic_budget_temperature', type=float, default=1.0, help='Temperature for semantic risk budget allocation')
  parser.add_argument('--semantic_drift_threshold', type=float, default=0.0, help='Allowed episode-average semantic drift before lambda increases')
  parser.add_argument('--semantic_dual_step_size', type=float, default=0.1, help='Dual feedback step size for semantic drift control')
  parser.add_argument('--semantic_lambda_max', type=float, default=10.0, help='Maximum per-episode semantic drift feedback multiplier')
  parser.add_argument('--semantic_sigma_min', type=float, default=1e-6, help='Minimum valid style standard deviation')
  parser.add_argument('--no_gnn', action='store_true', default=False, help='Disable GNN and fall back to prototype scoring')
  parser.add_argument('--disable_style_adv_generator', action='store_true', default=False, help='Disable style adversarial generation and train only on the original branch')

  parser.add_argument('--target_datasets', default='', type=str, help='Comma-separated target datasets for cross-domain evaluation')
  parser.add_argument('--target_dataset', default='', type=str, help='Single target dataset for pairwise source->target training/evaluation')
  parser.add_argument('--do_bscdfsl', type=int, default=0, help='Enable cross-domain evaluation')
  parser.add_argument('--skip_source_test', type=int, default=0, help='Skip source-domain novel evaluation after training')
  parser.add_argument('--require_warmup', type=int, default=0, help='Fail if warmup checkpoint is missing')
  parser.add_argument('--target_unlabeled_batch_size', default=64, type=int, help='Batch size for target unlabeled training branch')
  parser.add_argument('--target_ssl_weight', default=1.0, type=float, help='Weight of the target unlabeled consistency loss')
  parser.add_argument(
    '--target_ssl_mode',
    default='legacy_pseudo',
    choices=['legacy_pseudo', 'feature_consistency'],
    help='Target SSL objective: legacy source-class pseudo labels or label-free weak/strong feature consistency',
  )
  parser.add_argument('--target_ssl_temperature', default=0.5, type=float, help='Temperature used to sharpen target pseudo-labels')
  parser.add_argument('--target_ssl_ramp_epochs', default=60, type=int, help='Ramp-up epochs for target unlabeled consistency weight')
  parser.add_argument('--target_ssl_confidence_threshold', default=0.80, type=float, help='Confidence threshold for keeping target pseudo-labels')
  parser.add_argument('--target_ssl_max_entropy', default=0.75, type=float, help='Maximum normalized entropy allowed for target pseudo-labels')
  parser.add_argument('--target_ssl_view_threshold', default=0.65, type=float, help='Minimum weak/strong view agreement for target SSL')
  parser.add_argument('--target_ssl_feature_weight', default=0.25, type=float, help='Weight of feature-level weak/strong consistency in target SSL')
  parser.add_argument('--target_ssl_prototype_weight', default=0.25, type=float, help='Weight of target pseudo-prototype consistency in target SSL')
  parser.add_argument('--target_ssl_weight_floor', default=0.0, type=float, help='Lower bound for confidence-aware target SSL weight scaling')
  parser.add_argument('--eval_episode_relevance', type=int, default=0, help='Log episode-level prototype relevance diagnostics during evaluation')
  parser.add_argument('--eval_proto_refine', type=int, default=0, help='Enable inference-only episode-space prototype refinement during evaluation')
  parser.add_argument('--eval_proto_refine_alpha_1shot', default=8.0, type=float, help='Support anchor used by inference-only prototype refinement for 1-shot evaluation')
  parser.add_argument('--eval_proto_refine_alpha_5shot', default=4.0, type=float, help='Support anchor used by inference-only prototype refinement for 5-shot evaluation')
  parser.add_argument('--eval_proto_refine_affinity_threshold', default=0.55, type=float, help='Minimum prototype affinity required by inference-only refinement')
  parser.add_argument('--eval_proto_refine_margin_threshold', default=0.05, type=float, help='Minimum prototype top1-top2 margin required by inference-only refinement')
  parser.add_argument('--eval_proto_refine_min_selected', default=8, type=int, help='Minimum number of gate-passed target samples required to apply inference-only refinement')
  parser.add_argument('--eval_proto_refine_min_coverage', default=0.40, type=float, help='Minimum class coverage ratio required to apply inference-only refinement')
  parser.add_argument('--eval_proto_refine_affinity_threshold_1shot', default=0.70, type=float, help='1-shot prototype affinity threshold used by inference-only refinement')
  parser.add_argument('--eval_proto_refine_affinity_threshold_5shot', default=0.65, type=float, help='5-shot prototype affinity threshold used by inference-only refinement')
  parser.add_argument('--eval_proto_refine_margin_threshold_1shot', default=0.12, type=float, help='1-shot prototype margin threshold used by inference-only refinement')
  parser.add_argument('--eval_proto_refine_margin_threshold_5shot', default=0.10, type=float, help='5-shot prototype margin threshold used by inference-only refinement')
  parser.add_argument('--eval_proto_refine_min_selected_1shot', default=15, type=int, help='1-shot minimum gate-passed target sample count used by inference-only refinement')
  parser.add_argument('--eval_proto_refine_min_selected_5shot', default=12, type=int, help='5-shot minimum gate-passed target sample count used by inference-only refinement')
  parser.add_argument('--eval_proto_refine_min_coverage_1shot', default=0.60, type=float, help='1-shot minimum class coverage used by inference-only refinement')
  parser.add_argument('--eval_proto_refine_min_coverage_5shot', default=0.50, type=float, help='5-shot minimum class coverage used by inference-only refinement')
  parser.add_argument('--eval_proto_refine_topk_1shot', default=15, type=int, help='Maximum number of gate-passed target samples used by inference-only refinement in 1-shot')
  parser.add_argument('--eval_proto_refine_support_loo_guard', default=1, type=int, help='Reject inference-only prototype refinement when support prediction accuracy degrades')
  parser.add_argument('--eval_proto_refine_support_loo_tolerance', default=0.0, type=float, help='Allowed support prediction accuracy drop before rejecting inference-only refinement')
  parser.add_argument('--eval_proto_refine_support_margin_guard', default=1, type=int, help='Reject inference-only prototype refinement when support leave-one-out margin degrades')
  parser.add_argument('--eval_proto_refine_support_margin_tolerance', default=0.0, type=float, help='Allowed support leave-one-out margin drop before rejecting inference-only refinement')
  parser.add_argument('--eval_proto_refine_target_guard', default=1, type=int, help='Reject inference-only prototype refinement when target-side relevance degrades')
  parser.add_argument('--eval_proto_refine_target_affinity_tolerance', default=0.0, type=float, help='Allowed target affinity drop before rejecting inference-only refinement')
  parser.add_argument('--eval_proto_refine_target_margin_tolerance', default=0.0, type=float, help='Allowed target margin drop before rejecting inference-only refinement')
  parser.add_argument('--eval_proto_refine_target_affinity_gain_1shot', default=0.05, type=float, help='1-shot minimum target affinity gain required before accepting inference-only refinement')
  parser.add_argument('--eval_proto_refine_target_affinity_gain_5shot', default=0.02, type=float, help='5-shot minimum target affinity gain required before accepting inference-only refinement')
  parser.add_argument('--eval_proto_refine_target_margin_gain_1shot', default=0.05, type=float, help='1-shot minimum target margin gain required before accepting inference-only refinement')
  parser.add_argument('--eval_proto_refine_target_margin_gain_5shot', default=0.02, type=float, help='5-shot minimum target margin gain required before accepting inference-only refinement')

  parser.add_argument('--train_episodes', default=100, type=int, help='Training episodes per epoch')
  parser.add_argument('--val_episodes', default=100, type=int, help='Validation episodes per epoch')
  parser.add_argument('--n_episodes_test', default=1000, type=int, help='Evaluation episodes')
  parser.add_argument('--n_query', default=15, type=int, help='Query samples per class for episodic evaluation')
  parser.add_argument('--train_n_query', default=None, type=int,
                      help='Optional query samples per class during training; defaults to the legacy automatic rule')
  parser.add_argument('--shuffle_query_labels', action='store_true', help='Shuffle query labels during evaluation')
  parser.add_argument('--disable_progress_bar', action='store_true', help='Disable tqdm/progress bars')
  parser.add_argument('--resume_from', default='', type=str, help='Resume from a specific checkpoint path')

  parser.add_argument('--train_num_workers', default=8, type=int, help='Worker count for training data loading')
  parser.add_argument('--eval_num_workers', default=8, type=int, help='Worker count for validation/test data loading')
  parser.add_argument('--pin_memory', type=int, default=1, help='Enable DataLoader pin_memory')
  parser.add_argument('--persistent_workers', type=int, default=1, help='Keep workers alive across epochs')
  parser.add_argument('--prefetch_factor', default=2, type=int, help='DataLoader prefetch factor when num_workers>0')
  parser.add_argument('--feature_batch_size', default=64, type=int, help='Batch size for feature extraction')

  parser.add_argument('--distributed', type=int, default=0, help='Enable multi-process distributed execution')
  parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for torchrun launches')
  parser.add_argument('--dist_backend', type=str, default='nccl', help='Distributed backend')

  parser.add_argument('--finetune_epoch', default=50, type=int, help='Finetuning epochs for adaptation-based evaluation')
  parser.add_argument('--resume_dir', default='Pretrain', type=str, help='Legacy checkpoint directory name')


def parse_args(script):
  parser = argparse.ArgumentParser(description='few-shot script %s' % script)
  _add_common_args(parser)

  if script == 'train':
    parser.add_argument('--num_classes', default=64, type=int, help='Total classes for baseline softmax')
    parser.add_argument('--save_freq', default=100, type=int, help='Checkpoint save frequency')
    parser.add_argument('--target_set', default='cub', help='Legacy labeled target dataset')
    parser.add_argument('--target_num_label', default=5, type=int, help='Labeled target images per class')
    parser.add_argument('--start_epoch', default=0, type=int, help='Starting epoch')
    parser.add_argument('--stop_epoch', default=200, type=int, help='Stopping epoch')
    parser.add_argument('--resume', default='', type=str, help='Resume from existing experiment directory')
    parser.add_argument('--resume_epoch', default=-1, type=int, help='Resume epoch, -1 for latest numeric checkpoint')
    parser.add_argument('--warmup', default='gg3b0', type=str, help='Warmup checkpoint directory')
  elif script == 'test':
    parser.add_argument('--eval_seed', default=None, type=int, help='Optional fixed episode seed for reproducible ablations')
    parser.add_argument('--split', default='novel', help='base/val/novel')
    parser.add_argument('--save_epoch', default=-1, type=int, help='Checkpoint epoch to load, -1 for best model')
    parser.add_argument('--warmup', default='gg3bo', type=str, help='Legacy warmup arg for test compatibility')
    parser.add_argument('--stop_epoch', default=400, type=int, help='Stopping epoch placeholder for test compatibility')
  else:
    raise ValueError('Unknown script')

  args = parser.parse_args()
  args.data_dir = absolute_path(args.data_dir)
  args.save_dir = absolute_path(args.save_dir, base_dir=REPO_ROOT)
  if getattr(args, 'resume_from', ''):
    args.resume_from = absolute_path(args.resume_from, base_dir=REPO_ROOT)
  if getattr(args, 'text_guidance_path', ''):
    args.text_guidance_path = absolute_path(args.text_guidance_path, base_dir=REPO_ROOT)

  if not os.path.isdir(args.data_dir):
    parser.error(f'--data_dir does not exist or is not a directory: {args.data_dir}')
  os.makedirs(args.save_dir, exist_ok=True)
  return args


def get_assigned_file(checkpoint_dir, num):
  return os.path.join(checkpoint_dir, '{:d}.tar'.format(num))


def get_resume_file(checkpoint_dir, resume_epoch=-1):
  filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
  if len(filelist) == 0:
    return None

  numeric_files = []
  for path in filelist:
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem in {'best_model', 'last_epoch'}:
      continue
    if stem.isdigit():
      numeric_files.append(path)

  if resume_epoch != -1:
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(resume_epoch))
    return resume_file if os.path.isfile(resume_file) else None

  if numeric_files:
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in numeric_files])
    max_epoch = int(np.max(epochs))
    return os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))

  last_epoch = os.path.join(checkpoint_dir, 'last_epoch.tar')
  return last_epoch if os.path.isfile(last_epoch) else None


def get_best_file(checkpoint_dir):
  best_file = os.path.join(checkpoint_dir, 'best_model.tar')
  if os.path.isfile(best_file):
    return best_file
  return get_resume_file(checkpoint_dir)


def load_warmup_state(filename):
  if is_main_process():
    print('  load pre-trained model file: {}'.format(filename))
  warmup_resume_file = get_resume_file(filename)
  if is_main_process():
    print(' warmup_resume_file:', warmup_resume_file)
  if warmup_resume_file is None:
    raise ValueError('No pre-trained encoder file found!')
  tmp = torch.load(warmup_resume_file, weights_only=False)
  if tmp is None:
    raise ValueError('No pre-trained encoder file found!')

  state = tmp['state']
  state_keys = list(state.keys())
  for key in state_keys:
    if 'feature.' in key:
      newkey = key.replace('feature.', '')
      state[newkey] = state.pop(key)
    else:
      state.pop(key)
  return state
