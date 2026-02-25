"""Combine four training loss PDF plots into a single 2x2 PDF."""
import fitz  # PyMuPDF: pip install PyMuPDF
import matplotlib.pyplot as plt
import numpy as np

ROOT = r"c:\Users\EDY\Desktop\palm_fusion_vein"

def pdf_to_image(path):
    """Render the first page of a PDF to a numpy array."""
    doc = fitz.open(path)
    page = doc[0]
    pix = page.get_pixmap(dpi=300)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    doc.close()
    return img

pdfs = {
    "PolyU":  f"{ROOT}\\polyu_student_loss_curves.pdf",
    "CASIA":  f"{ROOT}\\CASIA_student_loss_curves.pdf",
    "Tongji": f"{ROOT}\\tongji_student_loss_curves.pdf",
    "CUMT":   f"{ROOT}\\CUMT_student_loss_curves.pdf",
}

order = ["PolyU", "CASIA", "Tongji", "CUMT"]
labels = ["(a)", "(b)", "(c)", "(d)"]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, name, label in zip(axes.flat, order, labels):
    img = pdf_to_image(pdfs[name])
    ax.imshow(img)
    ax.set_title(f"{label} {name}", fontsize=13, fontweight="bold", pad=6)
    ax.axis("off")

fig.tight_layout(pad=1.5)
out = f"{ROOT}\\combined_student_loss_curves.pdf"
fig.savefig(out, dpi=300, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
print(f"Saved to {out}")
plt.show()
