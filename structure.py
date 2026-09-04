import os
from pathlib import Path

root = Path(".").resolve()
max_depth = 4  # 根据需要调整层级

def walk(path: Path, prefix="", depth=0):
    if depth > max_depth:
        return
    entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    for i, p in enumerate(entries):
        connector = "└── " if i == len(entries) - 1 else "├── "
        print(f"{prefix}{connector}{p.name}")
        if p.is_dir() and not p.is_symlink():
            extension = "    " if i == len(entries) - 1 else "│   "
            walk(p, prefix + extension, depth + 1)

print(root)
walk(root)


