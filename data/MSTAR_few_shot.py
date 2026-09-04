"""
Placeholder loader for MSTAR (SAR) with DKD angle-based split (17° unlabeled / 15° eval).

Implement actual parsing when MSTAR is available. For now, instantiating the datamanagers
will raise a clear NotImplementedError to avoid silent misuse.
"""


class SimpleDataManager:
  def __init__(self, *args, **kwargs):
    raise NotImplementedError("MSTAR loader not implemented yet. Please add angle-based (17°/15°) handling.")

  def get_data_loader(self, *args, **kwargs):
    raise NotImplementedError("MSTAR loader not implemented yet.")


class SetDataManager(SimpleDataManager):
  pass
