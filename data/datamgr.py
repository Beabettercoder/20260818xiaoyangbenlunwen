# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate
import json
import os
import inspect
import numpy as np
import torch
import random
from PIL import Image
import torchvision.transforms as transforms
import data.additional_transforms as add_transforms
from data.augmentations_dkd import build_strong_transform, build_weak_transform
from data.dataset import SimpleDataset, SetDataset, MultiSetDataset, EpisodicBatchSampler, MultiEpisodicBatchSampler, RandomLabeledTargetDataset
from abc import abstractmethod
import importlib
import sys
from config_datasets import get_dataset_config, update_dataset_root
from utils.dataset_utils import load_dkd_split, get_dkd_split_path
from data.dataset import identity
from utils.path_utils import resolve_image_path

class TransformLoader:
  def __init__(self, image_size,
      normalize_param = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      jitter_param = dict(Brightness=0.2, Contrast=0.2, Color=0.2)):
    self.image_size = image_size
    self.normalize_param = normalize_param
    self.jitter_param = jitter_param

  def parse_transform(self, transform_type):
    if transform_type=='ImageJitter':
      method = add_transforms.ImageJitter( self.jitter_param )
      return method
    method = getattr(transforms, transform_type)

    if transform_type=='RandomResizedCrop':
      return method(self.image_size)
    elif transform_type=='CenterCrop':
      return method(self.image_size)
    elif transform_type=='Resize':
      return method([int(self.image_size*1.15), int(self.image_size*1.15)])
    elif transform_type=='Normalize':
      return method(**self.normalize_param )
    else:
      return method()

  def get_composed_transform(self, aug = False):
    if aug:
      transform_list = ['RandomResizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'RandomVerticalFlip', 'ToTensor', 'Normalize']
    else:
      transform_list = ['Resize','CenterCrop', 'ToTensor', 'Normalize']

    transform_funcs = [ self.parse_transform(x) for x in transform_list]
    transform = transforms.Compose(transform_funcs)
    return transform


# Inline dataset helpers (DKD splits / meta reuse)
class InlineSimpleDataset(torch.utils.data.Dataset):
  """Simple dataset that consumes an in-memory meta dict instead of a JSON file."""
  def __init__(self, meta, transform, target_transform=identity, data_root=None, dataset_name=None):
    self.meta = meta
    self.transform = transform
    self.target_transform = target_transform
    self.data_root = data_root
    self.dataset_name = dataset_name

  def __getitem__(self, i):
    image_path = resolve_image_path(
      self.meta['image_names'][i],
      data_root=self.data_root,
      dataset_name=self.dataset_name,
    )
    with Image.open(image_path) as source:
      img = source.convert('RGB')
    img = self.transform(img)
    target = self.target_transform(self.meta['image_labels'][i])
    return img, target

  def __len__(self):
    return len(self.meta['image_names'])


class InlineDualAugDataset(torch.utils.data.Dataset):
  """Dataset that returns weak/strong augmented views for unlabeled target training."""
  def __init__(self, meta, weak_transform, strong_transform, data_root=None, dataset_name=None):
    self.meta = meta
    self.weak_transform = weak_transform
    self.strong_transform = strong_transform
    self.data_root = data_root
    self.dataset_name = dataset_name

  def __getitem__(self, i):
    image_path = resolve_image_path(
      self.meta['image_names'][i],
      data_root=self.data_root,
      dataset_name=self.dataset_name,
    )
    with Image.open(image_path) as img:
      img = img.convert('RGB')
      weak = self.weak_transform(img)
      strong = self.strong_transform(img)
    return weak, strong

  def __len__(self):
    return len(self.meta['image_names'])


def _load_meta_json(json_path):
  with open(json_path, 'r') as handle:
    return json.load(handle)


def _resolve_dataset_config(dataset_name, data_root_override=None):
  if data_root_override:
    update_dataset_root(dataset_name, os.path.join(data_root_override, dataset_name))
  return get_dataset_config(dataset_name)


def load_dataset_meta(dataset_name, splits=("base", "val", "novel"), data_root=None):
  """Load and concatenate meta from available split JSONs."""
  cfg = _resolve_dataset_config(dataset_name, data_root_override=data_root)
  meta = {"image_names": [], "image_labels": []}
  found = []
  for split in splits:
    jpath = cfg["splits"].get(split)
    if jpath and os.path.isfile(jpath):
      data = _load_meta_json(jpath)
      meta["image_names"].extend(data.get("image_names", []))
      meta["image_labels"].extend(data.get("image_labels", []))
      found.append(split)
  if not found:
    raise FileNotFoundError(f"No split JSONs found for {dataset_name} under {cfg['root']}")
  return meta, found


def subset_meta(meta, indices):
  return {
    "image_names": [meta["image_names"][i] for i in indices],
    "image_labels": [meta["image_labels"][i] for i in indices],
  }


def build_full_dataset(dataset_name, image_size=224, data_root=None, splits=("base", "val", "novel")):
  """Return an InlineSimpleDataset covering the requested splits (default: all)."""
  meta, found_splits = load_dataset_meta(dataset_name, splits=splits, data_root=data_root)
  trans_loader = TransformLoader(image_size)
  dataset = InlineSimpleDataset(
    meta,
    trans_loader.get_composed_transform(aug=False),
    data_root=data_root,
    dataset_name=dataset_name,
  )
  return dataset, found_splits


def load_dkd_split_for_dataset(dataset_name, data_root=None):
  cfg = _resolve_dataset_config(dataset_name, data_root_override=data_root)
  split_path = get_dkd_split_path(dataset_name, cfg["root"])
  if os.path.isfile(split_path):
    return load_dkd_split(split_path), split_path
  return None, split_path


def build_dkd_subset(dataset_name, split_role="eval", image_size=224, data_root=None, dkd_split=None, splits=("base", "val", "novel")):
  """
  Build an InlineSimpleDataset for DKD split role ('unlabeled' or 'eval').
  If dkd_split is None, attempt to load from default path; otherwise expect (unlabeled, eval) tuple.
  """
  if split_role not in ("unlabeled", "eval"):
    raise ValueError("split_role must be 'unlabeled' or 'eval'")
  if dkd_split is None:
    dkd_split, _ = load_dkd_split_for_dataset(dataset_name, data_root=data_root)
  if dkd_split is None:
    return None

  unlabeled_idx, eval_idx = dkd_split
  selected = unlabeled_idx if split_role == "unlabeled" else eval_idx
  meta, _ = load_dataset_meta(dataset_name, splits=splits, data_root=data_root)
  filtered_meta = subset_meta(meta, selected)
  trans_loader = TransformLoader(image_size)
  return InlineSimpleDataset(
    filtered_meta,
    trans_loader.get_composed_transform(aug=False),
    data_root=data_root,
    dataset_name=dataset_name,
  )


def build_unlabeled_dataloader(
  dataset_name,
  image_size=224,
  batch_size=64,
  data_root=None,
  split="unlabeled",
  num_workers=4,
  pin_memory=True,
  persistent_workers=True,
  prefetch_factor=2,
):
  split_file = resolve_dataset_split_file(dataset_name, split=split, data_root=data_root)
  if split_file is None:
    return None

  meta = _load_meta_json(split_file)
  dataset = InlineDualAugDataset(
    meta,
    build_weak_transform(image_size),
    build_strong_transform(image_size),
    data_root=data_root,
    dataset_name=dataset_name,
  )
  data_loader_params = dict(
    batch_size=batch_size,
    shuffle=True,
    drop_last=False,
    worker_init_fn=_seed_worker,
  )
  data_loader_params.update(
    _build_loader_params(
      num_workers,
      pin_memory,
      persistent_workers,
      prefetch_factor,
    )
  )
  return torch.utils.data.DataLoader(dataset, **data_loader_params)




# added by fuyuqian in 2021 0107
class LabeledTargetDataset:
   def __init__(self, data_file,image_size, batch_size = 16, aug=True):
       with open(data_file, 'r') as f:
           self.meta = json.load(f)
       #print('len of labeled target data:', len(self.meta['image_names']))
       # define transform
       self.batch_size = batch_size
       self.trans_loader = TransformLoader(image_size)
       self.transform = self.trans_loader.get_composed_transform(aug)

   def get_epoch(self):
       # return random
       idx_list = [i for i in range(len(self.meta['image_names']))]
       selected_idx_list = random.sample(idx_list, self.batch_size)
       
       img_list = []
       img_label = []
     
       for idx in selected_idx_list:
           image_path = self.meta['image_names'][idx]
           image_label = self.meta['image_labels'][idx]
           img = Image.open(image_path).convert('RGB')
           img = self.transform(img)
           img_list.append(img)
           img_label.append(image_label)
       #print(img_label)
       img_list = torch.stack(img_list)
       #img_label = torch.stack(img_label)
       img_label = torch.LongTensor(img_label)
       #print('img_list:', img_list.size())
       #print('img_label:', img_label.size())
       return img_list, img_label



class DataManager:
  @abstractmethod
  def get_data_loader(self, data_file, aug):
    pass


class BoundDataManager(DataManager):
  """Bind a generic data source so callers can keep using get_data_loader(aug=False)."""
  def __init__(self, datamgr, data_source):
    self.datamgr = datamgr
    self.data_source = data_source

  def get_data_loader(self, data_file=None, aug=False):
    source = self.data_source if data_file is None else data_file
    return self.datamgr.get_data_loader(source, aug)


def _build_loader_params(num_workers, pin_memory, persistent_workers, prefetch_factor):
  params = dict(
    num_workers=num_workers,
    pin_memory=pin_memory,
  )
  if num_workers > 0:
    params["persistent_workers"] = persistent_workers
    if prefetch_factor is not None:
      params["prefetch_factor"] = prefetch_factor
  return params


def _seed_worker(worker_id):
  worker_seed = torch.initial_seed() % (2 ** 32)
  np.random.seed(worker_seed)
  random.seed(worker_seed)

class SimpleDataManager(DataManager):
  def __init__(self, image_size, batch_size, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2, data_root=None, dataset_name=None):
    super(SimpleDataManager, self).__init__()
    self.batch_size = batch_size
    self.trans_loader = TransformLoader(image_size)
    self.num_workers = num_workers
    self.pin_memory = pin_memory
    self.persistent_workers = persistent_workers
    self.prefetch_factor = prefetch_factor
    self.data_root = data_root
    self.dataset_name = dataset_name

  def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
    transform = self.trans_loader.get_composed_transform(aug)
    if isinstance(data_file, dict):
      dataset = InlineSimpleDataset(
        data_file,
        transform,
        data_root=self.data_root,
        dataset_name=self.dataset_name,
      )
    elif isinstance(data_file, InlineSimpleDataset):
      dataset = data_file
    else:
      dataset = SimpleDataset(
        data_file,
        transform,
        data_root=self.data_root,
        dataset_name=self.dataset_name,
      )
    data_loader_params = dict(
      batch_size=self.batch_size,
      shuffle=True,
      worker_init_fn=_seed_worker,
    )
    data_loader_params.update(
      _build_loader_params(
        self.num_workers,
        self.pin_memory,
        self.persistent_workers,
        self.prefetch_factor,
      )
    )
    data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

    return data_loader


# added in 20210108
class RandomLabeledTargetDataManager(DataManager):
  def __init__(self, image_size, batch_size, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2, data_root=None, dataset_name=None):
    super(RandomLabeledTargetDataManager, self).__init__()
    self.batch_size = batch_size
    self.trans_loader = TransformLoader(image_size)
    self.num_workers = num_workers
    self.pin_memory = pin_memory
    self.persistent_workers = persistent_workers
    self.prefetch_factor = prefetch_factor
    self.data_root = data_root
    self.dataset_name = dataset_name

  def get_data_loader(self, data_file, data_file_miniImagenet, aug): #parameters that would change on train/val set
    transform = self.trans_loader.get_composed_transform(aug)
    dataset = RandomLabeledTargetDataset(
      data_file,
      data_file_miniImagenet,
      transform,
      data_root=self.data_root,
      dataset_name=self.dataset_name,
    )
    data_loader_params = dict(
      batch_size=self.batch_size,
      shuffle=True,
      worker_init_fn=_seed_worker,
    )
    data_loader_params.update(
      _build_loader_params(
        self.num_workers,
        self.pin_memory,
        self.persistent_workers,
        self.prefetch_factor,
      )
    )
    data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

    return data_loader

class SetDataManager(DataManager):
  def __init__(
    self,
    image_size,
    n_way,
    n_support,
    n_query,
    n_eposide=100,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
    sampler_seed=0,
    world_size=1,
    rank=0,
    data_root=None,
    dataset_name=None,
  ):
    super(SetDataManager, self).__init__()
    self.image_size = image_size
    self.n_way = n_way
    self.batch_size = n_support + n_query
    self.n_eposide = n_eposide
    self.num_workers = num_workers
    self.pin_memory = pin_memory
    self.persistent_workers = persistent_workers
    self.prefetch_factor = prefetch_factor
    self.sampler_seed = sampler_seed
    self.world_size = world_size
    self.rank = rank
    self.data_root = data_root
    self.dataset_name = dataset_name

    self.trans_loader = TransformLoader(image_size)
    #print('datamgr:', 'SetDataManager:', 'n_way:', self.n_way, 'batch_size:', self.batch_size)

  def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
    transform = self.trans_loader.get_composed_transform(aug)
    if isinstance(data_file, list):
      dataset = MultiSetDataset(
        data_file,
        self.batch_size,
        transform,
        data_root=self.data_root,
        dataset_name=self.dataset_name,
      )
      sampler = MultiEpisodicBatchSampler(
        dataset.lens(),
        self.n_way,
        self.n_eposide,
        seed=self.sampler_seed,
        rank=self.rank,
        world_size=self.world_size,
      )
    else:
      dataset = SetDataset(
        data_file,
        self.batch_size,
        transform,
        data_root=self.data_root,
        dataset_name=self.dataset_name,
      )
      sampler = EpisodicBatchSampler(
        len(dataset),
        self.n_way,
        self.n_eposide,
        seed=self.sampler_seed,
        rank=self.rank,
        world_size=self.world_size,
      )
    data_loader_params = dict(
      batch_sampler=sampler,
      worker_init_fn=_seed_worker,
    )
    data_loader_params.update(
      _build_loader_params(
        self.num_workers,
        self.pin_memory,
        self.persistent_workers,
        self.prefetch_factor,
      )
    )
    data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
    return data_loader

# ================================================================
# Dataset registry helpers (few-shot datamgr lookup)
# ================================================================
_LOADER_CACHE = {}


def _get_loader_module(dataset_name, force_reload=False):
  cfg = get_dataset_config(dataset_name)
  module_name = cfg.get("loader")
  if not module_name:
    raise ValueError(f"No loader specified for dataset '{dataset_name}'")
  module_path = f"data.{module_name}"
  if force_reload:
    _LOADER_CACHE.pop(module_name, None)
    sys.modules.pop(module_path, None)
    sys.modules.pop(module_name, None)
  if module_name in _LOADER_CACHE and not force_reload:
    return _LOADER_CACHE[module_name]
  module = importlib.import_module(module_path)
  _LOADER_CACHE[module_name] = module
  return module


def _instantiate_with_supported_kwargs(cls, *args, **kwargs):
  """Instantiate legacy dataset datamgrs while ignoring unsupported kwargs."""
  signature = inspect.signature(cls.__init__)
  supported = {}
  for key, value in kwargs.items():
    if key in signature.parameters:
      supported[key] = value
  return cls(*args, **supported)


def build_simple_datamgr(
  dataset_name,
  image_size=224,
  batch_size=64,
  force_reload=False,
  num_workers=4,
  pin_memory=True,
  persistent_workers=True,
  prefetch_factor=2,
):
  module = _get_loader_module(dataset_name, force_reload=force_reload)
  if not hasattr(module, "SimpleDataManager"):
    raise ValueError(f"Dataset '{dataset_name}' loader has no SimpleDataManager")
  return _instantiate_with_supported_kwargs(
    module.SimpleDataManager,
    image_size,
    batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=persistent_workers,
    prefetch_factor=prefetch_factor,
  )


def build_set_datamgr(
  dataset_name,
  image_size=224,
  n_way=5,
  n_support=5,
  n_query=16,
  n_eposide=100,
  force_reload=False,
  num_workers=4,
  pin_memory=True,
  persistent_workers=True,
  prefetch_factor=2,
  sampler_seed=0,
  world_size=1,
  rank=0,
):
  module = _get_loader_module(dataset_name, force_reload=force_reload)
  if not hasattr(module, "SetDataManager"):
    raise ValueError(f"Dataset '{dataset_name}' loader has no SetDataManager")
  return _instantiate_with_supported_kwargs(
    module.SetDataManager,
    image_size,
    n_way=n_way,
    n_support=n_support,
    n_query=n_query,
    n_eposide=n_eposide,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=persistent_workers,
    prefetch_factor=prefetch_factor,
    sampler_seed=sampler_seed,
    world_size=world_size,
    rank=rank,
  )


def resolve_dataset_split_file(dataset_name, split="novel", data_root=None):
  if data_root is not None:
    direct_path = os.path.join(data_root, dataset_name, f"{split}.json")
    if os.path.isfile(direct_path):
      return direct_path

  try:
    cfg = _resolve_dataset_config(dataset_name, data_root_override=data_root)
  except Exception:
    return None

  split_path = cfg.get("splits", {}).get(split)
  if split_path and os.path.isfile(split_path):
    return split_path
  return None


def get_few_shot_datamgr(dataset_name, episodic=True, **kwargs):
  """Factory to create dataset-specific datamgrs from the registry."""
  data_root_override = kwargs.pop("data_root", None)
  requested_split = kwargs.pop("split", "novel")

  split_file = resolve_dataset_split_file(
    dataset_name,
    split=requested_split,
    data_root=data_root_override,
  )
  if split_file is not None:
    if episodic:
      if "n_way" not in kwargs or "n_support" not in kwargs:
        raise ValueError("n_way and n_support are required for episodic datamgrs")
      datamgr = SetDataManager(
        kwargs.get("image_size", 224),
        n_way=kwargs["n_way"],
        n_support=kwargs["n_support"],
        n_query=kwargs.get("n_query", 16),
        n_eposide=kwargs.get("n_eposide", 100),
        num_workers=kwargs.get("num_workers", 4),
        pin_memory=kwargs.get("pin_memory", True),
        persistent_workers=kwargs.get("persistent_workers", True),
        prefetch_factor=kwargs.get("prefetch_factor", 2),
        sampler_seed=kwargs.get("sampler_seed", 0),
        world_size=kwargs.get("world_size", 1),
        rank=kwargs.get("rank", 0),
        data_root=data_root_override,
        dataset_name=dataset_name,
      )
      return BoundDataManager(datamgr, split_file)

    datamgr = SimpleDataManager(
      kwargs.get("image_size", 224),
      kwargs.get("batch_size", 64),
      num_workers=kwargs.get("num_workers", 4),
      pin_memory=kwargs.get("pin_memory", True),
      persistent_workers=kwargs.get("persistent_workers", True),
      prefetch_factor=kwargs.get("prefetch_factor", 2),
      data_root=data_root_override,
      dataset_name=dataset_name,
    )
    return BoundDataManager(datamgr, split_file)

  force_reload = False
  if data_root_override:
    _LOADER_CACHE.clear()
    update_dataset_root(dataset_name, os.path.join(data_root_override, dataset_name))
    force_reload = True
  if episodic:
    if "n_way" not in kwargs or "n_support" not in kwargs:
      raise ValueError("n_way and n_support are required for episodic datamgrs")
    return build_set_datamgr(
      dataset_name,
      force_reload=force_reload,
      image_size=kwargs.get("image_size", 224),
      n_way=kwargs["n_way"],
      n_support=kwargs["n_support"],
      n_query=kwargs.get("n_query", 16),
      n_eposide=kwargs.get("n_eposide", 100),
      num_workers=kwargs.get("num_workers", 4),
      pin_memory=kwargs.get("pin_memory", True),
      persistent_workers=kwargs.get("persistent_workers", True),
      prefetch_factor=kwargs.get("prefetch_factor", 2),
      sampler_seed=kwargs.get("sampler_seed", 0),
      world_size=kwargs.get("world_size", 1),
      rank=kwargs.get("rank", 0),
    )
  return build_simple_datamgr(
    dataset_name,
    force_reload=force_reload,
    image_size=kwargs.get("image_size", 224),
    batch_size=kwargs.get("batch_size", 64),
    num_workers=kwargs.get("num_workers", 4),
    pin_memory=kwargs.get("pin_memory", True),
    persistent_workers=kwargs.get("persistent_workers", True),
    prefetch_factor=kwargs.get("prefetch_factor", 2),
  )

# Legacy block (kept commented)
'''

# added in 20210109
class RandomLabeledTargetSetDataManager(DataManager):
  def __init__(self, image_size, n_way, n_support, n_query, n_eposide=100):
    super(RandomLabeledTargetSetDataManager, self).__init__()
    self.image_size = image_size
    self.n_way = n_way
    self.batch_size = n_support + n_query
    self.n_eposide = n_eposide

    self.trans_loader = TransformLoader(image_size)

  def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
    transform = self.trans_loader.get_composed_transform(aug)
    if isinstance(data_file, list):
      dataset = MultiSetDataset( data_file , self.batch_size, transform )
      sampler = MultiEpisodicBatchSampler(dataset.lens(), self.n_way, self.n_eposide )
    else:
      dataset = SetDataset( data_file , self.batch_size, transform )
      sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide )
    data_loader_params = dict(batch_sampler = sampler,  num_workers=4)
    data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
    return data_loader
 '''
