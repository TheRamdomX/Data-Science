import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RUTA_SALIDA = os.path.join(BASE_DIR, "EDA_Resultados")

def guardar(fig, nombre):
    ruta = os.path.join(RUTA_SALIDA, nombre)
    fig.savefig(ruta, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Guardado: {ruta}")

print("\n" + "="*60)
print("COMPARACION FINAL -- TODOS LOS MODELOS")
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
all_metrics = []

for archivo in archivos:
    data = np.load(archivo, allow_pickle=True)
    labels = data["labels"]
    for label in labels:
        key = str(label).replace("-", "_")
        fpr = data[f"{key}_fpr"]
        tpr = data[f"{key}_tpr"]
        auc = float(data[f"{key}_auc"][0])
        all_curves.append({"label": str(label), "fpr": fpr, "tpr": tpr, "auc": auc})

        has_extra = f"{key}_precision" in data
        if has_extra:
            all_metrics.append({
                "label": str(label),
                "auc": auc,
                "precision": float(data[f"{key}_precision"][0]),
                "recall": float(data[f"{key}_recall"][0]),
                "f1": float(data[f"{key}_f1"][0]),
                "cm": data[f"{key}_cm"],
            })

all_curves.sort(key=lambda x: x["auc"], reverse=True)
all_metrics.sort(key=lambda x: x["auc"], reverse=True)

# =========================================================
# TABLA CONSOLA
# =========================================================

print(f"\n  {'Modelo':<12} {'ROC-AUC':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print(f"  {'-'*54}")
for m in all_metrics:
    print(f"  {m['label']:<12} {m['auc']:>10.4f} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")

# =========================================================
# 1. GRAFICO ROC COMBINADO
# =========================================================

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
ax.set_title("Comparacion ROC -- Todos los Modelos (undersampling, sin leakage)", fontsize=13)
ax.legend(loc="lower right", fontsize=8)
ax.grid(True, alpha=0.2)
fig.tight_layout()
guardar(fig, "comparacion_todos_roc.png")

# =========================================================
# 2. TABLA DE METRICAS (imagen)
# =========================================================

if all_metrics:
    labels_t   = [m["label"] for m in all_metrics]
    auc_vals   = [m["auc"] for m in all_metrics]
    prec_vals  = [m["precision"] for m in all_metrics]
    rec_vals   = [m["recall"] for m in all_metrics]
    f1_vals    = [m["f1"] for m in all_metrics]

    cell_text = []
    for m in all_metrics:
        cell_text.append([
            f"{m['auc']:.4f}",
            f"{m['precision']:.4f}",
            f"{m['recall']:.4f}",
            f"{m['f1']:.4f}",
        ])

    col_labels = ["ROC-AUC", "Precision (1)", "Recall (1)", "F1-Score (1)"]

    fig, ax = plt.subplots(figsize=(10, 0.6 * len(labels_t) + 2))
    ax.axis("off")
    ax.set_title("Comparacion de Metricas -- Todos los Modelos\n(clase positiva = oferta aceptada)",
                 fontsize=13, fontweight="bold", pad=20)

    table = ax.table(
        cellText=cell_text,
        rowLabels=labels_t,
        colLabels=col_labels,
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        elif col == -1:
            cell.set_facecolor("#D9E2F3")
            cell.set_text_props(fontweight="bold")
        else:
            val = float(cell_text[row - 1][col])
            col_vals = [float(cell_text[r][col]) for r in range(len(cell_text))]
            vmin, vmax = min(col_vals), max(col_vals)
            if vmax > vmin:
                norm = (val - vmin) / (vmax - vmin)
            else:
                norm = 0.5
            r_c = 1.0 - 0.3 * norm
            g_c = 0.85 + 0.15 * norm
            b_c = 1.0 - 0.3 * norm
            cell.set_facecolor((r_c, g_c, b_c))

    fig.tight_layout()
    guardar(fig, "comparacion_tabla_metricas.png")

    # =========================================================
    # 3. GRAFICO DE BARRAS COMPARATIVO
    # =========================================================

    x = np.arange(len(labels_t))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - 1.5*width, auc_vals,  width, label="ROC-AUC",  color="#4472C4")
    bars2 = ax.bar(x - 0.5*width, prec_vals, width, label="Precision", color="#ED7D31")
    bars3 = ax.bar(x + 0.5*width, rec_vals,  width, label="Recall",    color="#70AD47")
    bars4 = ax.bar(x + 1.5*width, f1_vals,   width, label="F1-Score",  color="#FFC000")

    ax.set_ylabel("Valor", fontsize=11)
    ax.set_title("Comparacion de Metricas por Modelo", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_t, rotation=30, ha="right")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    guardar(fig, "comparacion_barras_metricas.png")

    # =========================================================
    # 4. MATRICES DE CONFUSION COMBINADAS
    # =========================================================

    n_models = len(all_metrics)
    ncols = min(4, n_models)
    nrows = (n_models + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, m in enumerate(all_metrics):
        ax = axes[i]
        cm = m["cm"]
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        total = cm.sum()

        im = ax.imshow(cm, cmap="Blues", aspect="auto")

        for r in range(2):
            for c_idx in range(2):
                count = cm[r, c_idx]
                pct = 100.0 * count / total
                ax.text(c_idx, r, f"{count:,}\n({pct:.1f}%)",
                        ha="center", va="center", fontsize=9,
                        color="white" if count > cm.max() * 0.5 else "black")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["Real 0", "Real 1"])
        ax.set_title(f"{m['label']}\nAUC={m['auc']:.4f}", fontsize=10, fontweight="bold")

    for j in range(n_models, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Matrices de Confusion -- Todos los Modelos", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    guardar(fig, "comparacion_matrices_confusion.png")

    # =========================================================
    # 5. TABLA CONFUSION RESUMIDA (TP, FP, TN, FN)
    # =========================================================

    col_labels_cm = ["TN", "FP", "FN", "TP", "Accuracy"]
    cell_text_cm = []
    for m in all_metrics:
        cm = m["cm"]
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        acc = (tn + tp) / cm.sum()
        cell_text_cm.append([
            f"{tn:,}", f"{fp:,}", f"{fn:,}", f"{tp:,}", f"{acc:.4f}"
        ])

    fig, ax = plt.subplots(figsize=(12, 0.6 * len(labels_t) + 2))
    ax.axis("off")
    ax.set_title("Resumen Matrices de Confusion -- Todos los Modelos",
                 fontsize=13, fontweight="bold", pad=20)

    table = ax.table(
        cellText=cell_text_cm,
        rowLabels=labels_t,
        colLabels=col_labels_cm,
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        elif col == -1:
            cell.set_facecolor("#D9E2F3")
            cell.set_text_props(fontweight="bold")

    fig.tight_layout()
    guardar(fig, "comparacion_tabla_confusion.png")

else:
    print("\n  NOTA: No se encontraron metricas extra (precision/recall/cm).")
    print("  Re-ejecuta los scripts de modelos para generar las tablas comparativas.")

print("\n" + "="*60)
print("COMPARACION COMPLETA")
print("="*60)
