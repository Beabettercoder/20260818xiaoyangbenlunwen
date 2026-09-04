import math
import os

import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


_COALESCE_BUCKET_BYTES = 32 * 1024 * 1024


def is_dist_avail_and_initialized():
  return dist.is_available() and dist.is_initialized()


def get_world_size():
  if not is_dist_avail_and_initialized():
    return 1
  return dist.get_world_size()


def get_rank():
  if not is_dist_avail_and_initialized():
    return 0
  return dist.get_rank()


def is_main_process():
  return get_rank() == 0


def barrier():
  if is_dist_avail_and_initialized():
    dist.barrier()


def cleanup_distributed():
  if is_dist_avail_and_initialized():
    dist.destroy_process_group()


def unwrap_model(model):
  return model.module if hasattr(model, 'module') else model


def get_local_rank(args=None):
  if args is not None and getattr(args, 'local_rank', -1) >= 0:
    return args.local_rank
  for key in ('LOCAL_RANK', 'SLURM_LOCALID'):
    value = os.environ.get(key)
    if value is not None:
      return int(value)
  return 0


def _iter_tensor_buckets(tensors, bucket_cap_bytes=_COALESCE_BUCKET_BYTES):
  grouped = {}
  for tensor in tensors:
    if tensor is None or not torch.is_tensor(tensor):
      continue
    key = (tensor.device, tensor.dtype)
    grouped.setdefault(key, []).append(tensor)

  for bucket_tensors in grouped.values():
    bucket = []
    bucket_bytes = 0
    for tensor in bucket_tensors:
      tensor_bytes = tensor.numel() * tensor.element_size()
      if bucket and bucket_bytes + tensor_bytes > bucket_cap_bytes:
        yield bucket
        bucket = []
        bucket_bytes = 0
      bucket.append(tensor)
      bucket_bytes += tensor_bytes
    if bucket:
      yield bucket


def _broadcast_coalesced_(tensors, src=0):
  for bucket in _iter_tensor_buckets(tensors):
    flat = _flatten_dense_tensors(bucket)
    dist.broadcast(flat, src=src)
    for tensor, synced in zip(bucket, _unflatten_dense_tensors(flat, bucket)):
      tensor.copy_(synced)


def get_env_world_size(args=None):
  env_world_size = os.environ.get('WORLD_SIZE') or os.environ.get('SLURM_NTASKS')
  if env_world_size is not None:
    return int(env_world_size)
  if args is not None and getattr(args, 'distributed', 0):
    return max(1, torch.cuda.device_count())
  return 1


def setup_distributed(args):
  world_size = get_env_world_size(args)
  local_rank = get_local_rank(args)
  use_distributed = bool(getattr(args, 'distributed', 0)) or world_size > 1
  has_launcher_env = os.environ.get('WORLD_SIZE') is not None or os.environ.get('SLURM_NTASKS') is not None

  if torch.cuda.is_available():
    if use_distributed:
      torch.cuda.set_device(local_rank)
      device = torch.device('cuda', local_rank)
    else:
      device = torch.device('cuda')
  else:
    device = torch.device('cpu')

  rank = 0
  if use_distributed and not is_dist_avail_and_initialized():
    if not has_launcher_env:
      raise RuntimeError(
        'Distributed mode requested, but launcher env is missing. '
        'Use torchrun --nproc_per_node=<num_gpus> ... --distributed 1'
      )
    backend = getattr(args, 'dist_backend', 'nccl')
    if backend == 'nccl' and not torch.cuda.is_available():
      backend = 'gloo'
    dist.init_process_group(backend=backend, init_method='env://')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
  elif is_dist_avail_and_initialized():
    rank = dist.get_rank()
    world_size = dist.get_world_size()

  args.distributed = int(use_distributed)
  args.rank = rank
  args.world_size = world_size
  args.local_rank = local_rank
  args.device = device
  return device


def average_gradients(model):
  if not is_dist_avail_and_initialized():
    return
  world_size = float(get_world_size())
  grads = [param.grad.data for param in model.parameters() if param.grad is not None]
  for bucket in _iter_tensor_buckets(grads):
    flat = _flatten_dense_tensors(bucket)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat /= world_size
    for grad, synced in zip(bucket, _unflatten_dense_tensors(flat, bucket)):
      grad.copy_(synced)


def sync_model_state(model):
  if not is_dist_avail_and_initialized():
    return
  tensors = [tensor for tensor in model.state_dict().values() if torch.is_tensor(tensor)]
  _broadcast_coalesced_(tensors, src=0)


def sync_model_buffers(model):
  if not is_dist_avail_and_initialized():
    return
  buffers = [buffer for _, buffer in model.named_buffers() if torch.is_tensor(buffer)]
  _broadcast_coalesced_(buffers, src=0)


def reduce_scalar(value, device, average=True):
  if not is_dist_avail_and_initialized():
    return float(value)
  tensor = torch.tensor(float(value), device=device)
  dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
  if average:
    tensor /= get_world_size()
  return tensor.item()


def shard_episode_count(total_episodes, world_size):
  if world_size <= 1:
    return total_episodes
  return int(math.ceil(total_episodes / float(world_size)))
