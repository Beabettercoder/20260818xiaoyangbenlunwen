try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def progress_range(iterable, desc=None):
    """Wrap an iterable with tqdm progress bar if available."""
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, ncols=100)


__all__ = ["progress_range"]
