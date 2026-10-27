"""Make a contact sheet (8 frames + joint plots) of one recorded episode."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

path = sys.argv[1]
d = np.load(path)
imgs, st = d["images"], d["states"]
print(path)
print("instruction:", str(d["instruction"]))
print("images:", imgs.shape, imgs.dtype, " states:", st.shape, " sec:", float(d["sec"]))
idx = np.linspace(0, len(imgs) - 1, 8).astype(int)
fig, axes = plt.subplots(2, 5, figsize=(16, 6.5))
for ax, i in zip(axes.flat[:8], idx):
    ax.imshow(imgs[i])
    ax.set_title(f"t={i / 10:.1f}s")
    ax.axis("off")
t = np.arange(len(st)) / 10
axes.flat[8].plot(t, st[:, :7])
axes.flat[8].set_title("arm joints [rad]")
axes.flat[9].plot(t, st[:, 7])
axes.flat[9].set_title("finger [m]")
fig.suptitle(str(d["instruction"]))
fig.tight_layout()
out = path.replace(".npz", ".png")
fig.savefig(out, dpi=70)
print("saved:", out)
