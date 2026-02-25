"""Plot CUMT student network training/validation loss curves (first 120 epochs)."""
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

def load(path, max_epoch=120):
    with open(path, "r") as f:
        data = json.load(f)
    epochs = [r[1] for r in data if r[1] <= max_epoch]
    losses = [r[2] for r in data if r[1] <= max_epoch]
    return epochs, losses

ep_train, loss_train = load(f"{ROOT}\\polyu_S_T.json")
ep_val, loss_val = load(f"{ROOT}\\polyu_S_V.json")

fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(ep_train, loss_train, color="#1f77b4", linewidth=1.8, label="Training Loss")
ax.plot(ep_val, loss_val, color="#ff7f0e", linewidth=1.8, label="Validation Loss")

ax.set_xlabel("Epoch", fontsize=13)
ax.set_ylabel("Loss", fontsize=13)
ax.legend(loc="upper right", fontsize=10, frameon=True, edgecolor="gray")
ax.set_xlim(0, 120)
ax.set_ylim(bottom=0)
ax.grid(False)

fig.subplots_adjust(bottom=0.22)
fig.text(0.5, 0.06,
         "Training and validation loss curves of PolyU",
         ha="center", fontsize=11, fontstyle="normal")

out = f"{ROOT}\\polyu_student_loss_curves.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print(f"Saved to {out}")
plt.show()
