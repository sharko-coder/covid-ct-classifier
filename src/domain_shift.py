import random
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

random.seed(0)
SETS = {
    "Kaggle COVID": "data/raw/COVID",
    "Kaggle non-COVID": "data/raw/non-COVID",
    "External COVID": "data/external/COVID",
    "External non-COVID": "data/external/non-COVID",
}
EXTS = {".png", ".jpg", ".jpeg"}

fig, axes = plt.subplots(4, 6, figsize=(16, 11))
for r, (title, folder) in enumerate(SETS.items()):
    files = sorted(f for f in Path(folder).iterdir() if f.suffix.lower() in EXTS)

    sample = random.sample(files, min(50, len(files)))
    sizes = [Image.open(f).size for f in sample]
    bright = [np.array(Image.open(f).convert("L")).mean() for f in sample]
    print(f"{title:20s} n={len(files):4d} | mean size "
          f"{np.mean([s[0] for s in sizes]):.0f}x{np.mean([s[1] for s in sizes]):.0f} "
          f"| mean brightness {np.mean(bright):.0f}/255")

    for c, f in enumerate(random.sample(files, 6)):
        ax = axes[r, c]
        ax.imshow(Image.open(f).convert("RGB"))
        ax.axis("off")
        if c == 0:
            ax.set_title(title, loc="left", fontsize=12)

plt.tight_layout()
plt.savefig("outputs/domain_shift.png", dpi=120, bbox_inches="tight")
print("Saved outputs/domain_shift.png")