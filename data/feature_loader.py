import torch
import numpy as np
import h5py
import time


def open_hdf5_file(filename, mode="r", retries=5, sleep_s=0.5, **kwargs):
  """Open HDF5 files robustly on shared filesystems that don't like file locking."""
  errors = []
  for attempt in range(retries):
    try:
      try:
        return h5py.File(filename, mode, locking=False, **kwargs)
      except TypeError:
        return h5py.File(filename, mode, **kwargs)
    except OSError as exc:
      errors.append(str(exc))
      if attempt + 1 >= retries:
        break
      time.sleep(sleep_s)
  raise OSError(
    f"Failed to open HDF5 file after {retries} attempts: {filename} | last_error={errors[-1]}"
  )

class SimpleHDF5Dataset:
  def __init__(self, file_handle = None):
    if file_handle == None:
      self.f = ''
      self.all_feats_dset = []
      self.all_labels = []
      self.total = 0
    else:
      self.f = file_handle
      self.all_feats_dset = self.f['all_feats'][...]
      self.all_labels = self.f['all_labels'][...]
      self.total = self.f['count'][0]
  def __getitem__(self, i):
    return torch.Tensor(self.all_feats_dset[i,:]), int(self.all_labels[i])

  def __len__(self):
    return self.total


def init_loader(filename):
  with open_hdf5_file(filename, 'r') as f:
    fileset = SimpleHDF5Dataset(f)

  feats = fileset.all_feats_dset
  labels = fileset.all_labels
  while np.sum(feats[-1]) == 0:
    feats  = np.delete(feats,-1,axis = 0)
    labels = np.delete(labels,-1,axis = 0)

  class_list = np.unique(np.array(labels)).tolist()
  inds = range(len(labels))

  cl_data_file = {}
  for cl in class_list:
    cl_data_file[cl] = []
  for ind in inds:
    cl_data_file[labels[ind]].append( feats[ind])

  return cl_data_file
