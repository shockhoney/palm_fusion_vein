"""Plot CUMT training loss curves (Palm / Vein / Fusion) in one figure."""
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.linewidth": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

ROOT = r"c:\Users\EDY\Desktop\palm_fusion_vein\json_files"

def load(path):
    with open(path, "r") as f:
        data = json.load(f)
    epochs = [r[1] for r in data]
    losses = [r[2] for r in data]
    return epochs, losses

ep_palm, loss_palm = load(f"{ROOT}\\CASIA_palm_train_loss.json")
ep_vein, loss_vein = load(f"{ROOT}\\CASIA_vein_train_loss.json")
ep_fuse, loss_fuse = load(f"{ROOT}\\CASIA_fusion_train_loss.json")

fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(ep_palm, loss_palm, color="#1f77b4", linewidth=1.8, label="Palmprint Training Loss")
ax.plot(ep_vein, loss_vein, color="#ff7f0e", linewidth=1.8, label="Palm-vein Training Loss")
ax.plot(ep_fuse, loss_fuse, color="#2ca02c", linewidth=1.8, label="Fusion Training Loss")

ax.set_xlabel("Epoch", fontsize=13)
ax.set_ylabel("Loss", fontsize=13)
ax.legend(loc="upper right", fontsize=10, frameon=True, edgecolor="gray")
ax.set_xlim(0, max(len(ep_palm), len(ep_vein), len(ep_fuse)) - 1)
ax.set_ylim(bottom=0)
ax.grid(False)

fig.subplots_adjust(bottom=0.22)
fig.text(0.5, 0.06, "Training loss curves of CASIA",
         ha="center", fontsize=11, fontstyle="normal")

out = f"{ROOT}\\CASIA_train_loss_curves.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print(f"Saved to {out}")
plt.show()
