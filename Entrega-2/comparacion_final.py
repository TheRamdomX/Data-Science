import numpy as np
import os
import glob
import matplotlib.pyplot as plt

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RUTA_SALIDA = os.path.join(BASE_DIR, "EDA_Resultados")

print("\n" + "="*60)
print("COMPARACIÓN FINAL — TODOS LOS MODELOS")
print("="*60)

archivos = sorted(glob.glob(os.path.join(RUTA_SALIDA, "resultados_*.npz")))

if not archivos:
    print("  ERROR: No se encontraron archivos resultados_*.npz")
    print("  Ejecuta primero: modelo.py, modelo_gb.py, modelo_nn.py, modelo_knn_svm.py")
    exit(1)

print(f"  Archivos encontrados: {len(archivos)}")
for a in archivos:
    print(f"    - {os.path.basename(a)}")

all_curves = []

for archivo in archivos:
    data = np.load(archivo, allow_pickle=True)
    labels = data["labels"]
    for label in labels:
        key = str(label).replace("-", "_")
        fpr = data[f"{key}_fpr"]
        tpr = data[f"{key}_tpr"]
        auc = float(data[f"{key}_auc"][0])
        all_curves.append({"label": str(label), "fpr": fpr, "tpr": tpr, "auc": auc})

all_curves.sort(key=lambda x: x["auc"], reverse=True)

print(f"\n  {'Modelo':<12} {'ROC-AUC':>10}")
print(f"  {'-'*24}")
for c in all_curves:
    print(f"  {c['label']:<12} {c['auc']:>10.4f}")

colors_a = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
colors_b = ["#aec7e8", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94"]

fig, ax = plt.subplots(figsize=(10, 8))

color_idx = 0
seen_models = {}
for c in all_curves:
    base_model = c["label"].split("-")[0]
    is_b = c["label"].endswith("-B")

    if base_model not in seen_models:
        seen_models[base_model] = color_idx
        color_idx += 1

    ci = seen_models[base_model]
    color = colors_b[ci % len(colors_b)] if is_b else colors_a[ci % len(colors_a)]
    style = "--" if is_b else "-"
    width = 1.2 if is_b else 2.0

    ax.plot(c["fpr"], c["tpr"], color=color, linestyle=style, linewidth=width,
            label=f"{c['label']} (AUC={c['auc']:.4f})")

ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.8, label="Random (AUC=0.5)")
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("Comparación ROC — Todos los Modelos (undersampling, sin leakage)", fontsize=13)
ax.legend(loc="lower right", fontsize=8)
ax.grid(True, alpha=0.2)
fig.tight_layout()

ruta = os.path.join(RUTA_SALIDA, "comparacion_todos_roc.png")
fig.savefig(ruta, bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"\n  Guardado: {ruta}")

print("\n" + "="*60)
print("COMPARACIÓN COMPLETA")
print("="*60)
