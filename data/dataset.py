# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import os
import random
from utils.path_utils import dataset_context, resolve_image_path


def identity(x):
  return x


def _normalize_image_path(image_path, data_root=None, dataset_name=None):
  return resolve_image_path(image_path, data_root=data_root, dataset_name=dataset_name)


def _sample_paths(paths, sample_size):
  if len(paths) >= sample_size:
    return random.sample(paths, sample_size)
  return random.choices(paths, k=sample_size)


def _load_image_rgb(image_path, transform, data_root=None, dataset_name=None):
  normalized = _normalize_image_path(
    image_path,
    data_root=data_root,
    dataset_name=dataset_name,
  )
  with Image.open(normalized) as img:
    img = img.convert('RGB')
    return transform(img)


class SimpleDataset:
  def __init__(self, data_file, transform, target_transform=identity, data_root=None, dataset_name=None):
    with open(data_file, 'r') as f:
      self.meta = json.load(f)
    self.data_root, self.dataset_name = dataset_context(
      data_file,
      data_root=data_root,
      dataset_name=dataset_name,
    )
    self.transform = transform
    self.target_transform = target_transform

  def __getitem__(self,i):
    image_path = self.meta['image_names'][i]
    img = _load_image_rgb(
      image_path,
      self.transform,
      data_root=self.data_root,
      dataset_name=self.dataset_name,
    )
    target = self.target_transform(self.meta['image_labels'][i])
    return img, target

  def __len__(self):
    return len(self.meta['image_names'])


# added by fuyuqian in 20210108
class RandomLabeledTargetDataset:
  def __init__(self, data_file,data_file_miniImagenet, transform, target_transform=identity, data_root=None, dataset_name=None):
    with open(data_file, 'r') as f:
      self.meta = json.load(f)
    with open(data_file_miniImagenet, 'r') as f_miniI:
      self.meta_miniImagenet = json.load(f_miniI)
    self.data_root, self.dataset_name = dataset_context(
      data_file,
      data_root=data_root,
      dataset_name=dataset_name,
    )
    self.transform = transform
    self.target_transform = target_transform

  def __getitem__(self,i):
    idx = random.randint(0, len(self.meta['image_names'])-1)
    image_path = self.meta['image_names'][idx]
    img = _load_image_rgb(
      image_path,
      self.transform,
      data_root=self.data_root,
      dataset_name=self.dataset_name,
    )
    target = self.target_transform(self.meta['image_labels'][idx])
    return img, target

  def __len__(self):
    #return len(self.meta['image_names'])
    return len(self.meta_miniImagenet['image_names'])


class SetDataset:
  def __init__(self, data_file, batch_size, transform, data_root=None, dataset_name=None):
    if isinstance(data_file, dict):
      self.meta = data_file
    else:
      with open(data_file, 'r') as f:
        self.meta = json.load(f)
    self.data_root, self.dataset_name = dataset_context(
      data_file,
      data_root=data_root,
      dataset_name=dataset_name,
    )

    self.cl_list = np.unique(self.meta['image_labels']).tolist()
    #print('dataset:', 'SetDataset:', 'cl_list:', self.cl_list)

    self.sub_meta = {}
    for cl in self.cl_list:
      self.sub_meta[cl] = []

    for x,y in zip(self.meta['image_names'],self.meta['image_labels']):
      self.sub_meta[y].append(x)

    #print('dataset:', 'SetDataset:', 'sub_meta:', len(self.sub_meta))
    #for i in range(len(self.sub_meta)):
        #print(i, len(self.sub_meta[i]))


    self.batch_size = batch_size
    self.transform = transform

  def __getitem__(self,i):
    cl = self.cl_list[i]
    sampled_paths = _sample_paths(self.sub_meta[cl], self.batch_size)
    images = [
      _load_image_rgb(
        path,
        self.transform,
        data_root=self.data_root,
        dataset_name=self.dataset_name,
      )
      for path in sampled_paths
    ]
    labels = torch.full((self.batch_size,), int(cl), dtype=torch.long)
    return torch.stack(images), labels

  def __len__(self):
    #print('dataset:', 'SetDataset:', 'len:', len(self.cl_list))
    return len(self.cl_list)


class MultiSetDataset:
  def __init__(self, data_files, batch_size, transform, data_root=None, dataset_name=None):
    self.cl_list = np.array([])
    self.n_classes = []
    self.class_entries = []
    self.batch_size = batch_size
    self.transform = transform
    self.data_root = data_root
    self.dataset_name = dataset_name
    for data_file in data_files:
      with open(data_file, 'r') as f:
        meta = json.load(f)
      cl_list = np.unique(meta['image_labels']).tolist()
      self.cl_list = np.concatenate((self.cl_list, cl_list))

      sub_meta = {}
      for cl in cl_list:
        sub_meta[cl] = []

      for x,y in zip(meta['image_names'], meta['image_labels']):
        sub_meta[y].append(x)

      for cl in cl_list:
        self.class_entries.append((cl, sub_meta[cl]))
      self.n_classes.append(len(cl_list))

  def __getitem__(self,i):
    cl, paths = self.class_entries[i]
    sampled_paths = _sample_paths(paths, self.batch_size)
    images = [
      _load_image_rgb(
        path,
        self.transform,
        data_root=self.data_root,
        dataset_name=self.dataset_name,
      )
      for path in sampled_paths
    ]
    labels = torch.full((self.batch_size,), int(cl), dtype=torch.long)
    return torch.stack(images), labels

  def __len__(self):
    return len(self.cl_list)

  def lens(self):
    return self.n_classes


class SubDataset:
  def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity, min_size=50, data_root=None, dataset_name=None):
    self.sub_meta = sub_meta
    self.cl = cl
    self.transform = transform
    self.target_transform = target_transform
    self.data_root = data_root
    self.dataset_name = dataset_name
    #print('dataset:', 'SubDatset:', 'sub_meta:', self.sub_meta)
    if len(self.sub_meta) < min_size:
      #print('dataset:', 'SubDataset:', 'len of self_meta:', len(self.sub_meta),' < 50')
      idxs = [i % len(self.sub_meta) for i in range(min_size)]
      #print('dataset:', 'SubDataset:', 'idxs:', idxs)
      self.sub_meta = np.array(self.sub_meta)[idxs].tolist()
      #print('dataset:', 'SubDataset:', 'sub_meat:', self.sub_meta)

  def __getitem__(self,i):
    image_path = self.sub_meta[i]
    img = _load_image_rgb(
      image_path,
      self.transform,
      data_root=self.data_root,
      dataset_name=self.dataset_name,
    )
    target = self.target_transform(self.cl)
    #print('img:',img.size(), 'target:', target)
    return img, target

  def __len__(self):
    return len(self.sub_meta)


class EpisodicBatchSampler(object):
  def __init__(self, n_classes, n_way, n_episodes, seed=0, rank=0, world_size=1):
    self.n_classes = n_classes
    self.n_way = n_way
    self.n_episodes = n_episodes
    self.seed = seed
    self.rank = rank
    self.world_size = max(1, world_size)
    self.epoch = 0

  def set_epoch(self, epoch):
    self.epoch = epoch

  def _local_episode_count(self):
    return int(np.ceil(self.n_episodes / float(self.world_size)))

  def __len__(self):
    return self._local_episode_count()

  def __iter__(self):
    generator = torch.Generator()
    generator.manual_seed(self.seed + self.epoch * 100003 + self.rank * 1009)
    for _ in range(self._local_episode_count()):
      yield torch.randperm(self.n_classes, generator=generator)[:self.n_way]


class MultiEpisodicBatchSampler(object):
  def __init__(self, n_classes, n_way, n_episodes, seed=0, rank=0, world_size=1):
    self.n_classes = n_classes
    self.n_way = n_way
    self.n_episodes = n_episodes
    self.n_domains = len(n_classes)
    self.seed = seed
    self.rank = rank
    self.world_size = max(1, world_size)
    self.epoch = 0

  def set_epoch(self, epoch):
    self.epoch = epoch

  def _local_episode_count(self):
    return int(np.ceil(self.n_episodes / float(self.world_size)))

  def __len__(self):
    return self._local_episode_count()

  def __iter__(self):
    local_episode_count = self._local_episode_count()
    domain_list = [i % self.n_domains for i in range(local_episode_count)]
    rng = random.Random(self.seed + self.epoch * 100003 + self.rank * 1009)
    rng.shuffle(domain_list)
    generator = torch.Generator()
    generator.manual_seed(self.seed + self.epoch * 200003 + self.rank * 2009)
    for i in range(local_episode_count):
      domain_idx = domain_list[i]
      start_idx = sum(self.n_classes[:domain_idx])
      yield torch.randperm(self.n_classes[domain_idx], generator=generator)[:self.n_way] + start_idx
