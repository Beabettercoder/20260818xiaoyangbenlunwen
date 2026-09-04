import os

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def plot_curve(x, y, xlabel, ylabel, title, save_path):
    if plt is None:
        print(f"[plot_utils] matplotlib not available, skip plotting {save_path}")
        return
    ensure_dir(os.path.dirname(save_path))
    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_bar(labels, values, xlabel, ylabel, title, save_path):
    if plt is None:
        print(f"[plot_utils] matplotlib not available, skip plotting {save_path}")
        return
    ensure_dir(os.path.dirname(save_path))
    plt.figure()
    plt.bar(labels, values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


__all__ = ["plot_curve", "plot_bar", "ensure_dir"]
