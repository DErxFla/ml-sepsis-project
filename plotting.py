# =========================
# plots.py  (general utils)
# =========================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay, balanced_accuracy_score, f1_score, log_loss


def plot_counts(series: pd.Series, title: str, rotate_xticks: int = 45):
    """Bar plot of value counts, including NaN."""
    vc = series.value_counts(dropna=False)
    plt.figure()
    vc.plot(kind="bar")
    plt.title(title)
    plt.ylabel("count")
    plt.xticks(rotation=rotate_xticks, ha="right")
    plt.tight_layout()
    plt.close()

def plot_counts_ax(series: pd.Series, title: str, ax, rotate_xticks: int = 45):
    vc = series.value_counts(dropna=False)
    vc.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("count")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=rotate_xticks)

def plot_missingness_bar_ax(df, cols, title, ax):
    miss = df[cols].isna().mean().sort_values(ascending=False)
    miss.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("fraction missing")
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=45)

def plot_missingness_hist_ax(df, cols, title, ax, bins=20):
    per_sample = df[cols].isna().mean(axis=1)
    ax.hist(per_sample.dropna(), bins=bins)
    ax.set_title(title)
    ax.set_xlabel("fraction missing (per sample)")
    ax.set_ylabel("count")
    ax.set_xlim(0, 1)
    return per_sample

def plot_expression_distributions(X: pd.DataFrame, title_prefix="Expression"):
    """
    Overall distribution and per-sample medians.
    X: samples x genes
    """
    vals = X.values.ravel()
    vals = vals[~np.isnan(vals)]

    plt.figure()
    plt.hist(vals, bins=60)
    plt.title(f"{title_prefix}: all values distribution")
    plt.xlabel("expression")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()

    med = X.median(axis=1, skipna=True)
    plt.figure()
    plt.hist(med, bins=40)
    plt.title(f"{title_prefix}: per-sample median expression")
    plt.xlabel("median expression")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()

    return med


from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

def plot_pca_ax(
    X: pd.DataFrame,
    labels: pd.Series,
    title: str,
    ax,
    scale: bool = True,
    impute: bool = True,
    dropna_labels: bool = True,
    alpha: float = 0.85,
    point_size: int = 22,
    min_samples: int = 3,
    debug: bool = True
):
    """
    Same as plot_pca, but draws on a provided matplotlib axis (for subplots).
    Returns (pc_df, pca) or (None, None) if skipped.
    """

    # --- Align labels to X samples ---
    labels = labels.reindex(X.index)

    # --- Decide which samples to include ---
    if dropna_labels:
        idx = labels.dropna().index
    else:
        idx = X.index

    if debug:
        print(f"[plot_pca_ax] title={title}")
        print(f"[plot_pca_ax] X samples={X.shape[0]}, genes={X.shape[1]}")
        print(f"[plot_pca_ax] labels non-missing={(~labels.isna()).sum()} / {len(labels)}")
        print(f"[plot_pca_ax] samples kept after filtering={len(idx)}")
        overlap = X.index.intersection(labels.index)
        print(f"[plot_pca_ax] index overlap X ∩ labels = {len(overlap)}")

    # --- Guardrail ---
    if len(idx) < min_samples:
        if debug:
            print(f"[plot_pca_ax] SKIP: need >= {min_samples} samples, got {len(idx)}.")
            print("[plot_pca_ax] label value counts (incl NaN):")
            print(labels.value_counts(dropna=False).head(10))
        ax.set_title(title + " (SKIPPED)")
        ax.axis("off")
        return None, None

    X_use = X.loc[idx]
    lab = labels.loc[idx]

    # --- Impute ---
    if impute:
        X_vals = SimpleImputer(strategy="median").fit_transform(X_use)
    else:
        X_vals = X_use.values

    # --- Scale ---
    if scale:
        X_vals = StandardScaler().fit_transform(X_vals)

    # --- PCA ---
    pca = PCA(n_components=2, random_state=0)
    pcs = pca.fit_transform(X_vals)

    pc_df = pd.DataFrame(pcs, index=X_use.index, columns=["PC1", "PC2"])
    exp = pca.explained_variance_ratio_ * 100

    # --- Plot by groups ---
    lab_plot = lab.astype("object").where(~lab.isna(), "NaN")

    for g in pd.unique(lab_plot):
        idxg = lab_plot[lab_plot == g].index
        ax.scatter(
            pc_df.loc[idxg, "PC1"],
            pc_df.loc[idxg, "PC2"],
            s=point_size,
            alpha=alpha,
            label=str(g)
        )

    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({exp[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({exp[1]:.1f}%)")
    ax.legend(fontsize=8)

    return pc_df, pca

def plot_cm_and_optional_roc(y_true, y_pred, y_proba, class_names, title_prefix="Model"):
    """
    Plots:
      1) Confusion matrix (counts)
      2) Confusion matrix (normalized by true class)
      3) ROC curve (macro-average OvR) if probabilities are provided

    y_proba:
      - shape (n_samples, n_classes) for multiclass probabilities
      - can be None if model doesn't support predict_proba
    """
    ncols = 3 if y_proba is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6*ncols, 4), constrained_layout=True)

    # --- Confusion matrix: counts ---
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title(f"{title_prefix} — Confusion Matrix (Counts)")

    # --- Confusion matrix: normalized (row-wise) ---
    disp2 = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        labels=class_names,
        display_labels=class_names,
        normalize="true",   # recall per class
        cmap="Blues",
        colorbar=False,
        ax=axes[1]
    )
    axes[1].set_title(f"{title_prefix} — Confusion Matrix (Normalized)")

    # --- Macro-average ROC (OvR), if probabilities are available ---
    if y_proba is not None:
        # Binarize true labels for OvR ROC-AUC
        y_true_bin = label_binarize(y_true, classes=class_names)

        # Compute macro-average ROC curve
        fpr = dict()
        tpr = dict()
        for i, cls in enumerate(class_names):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])

        # Macro-average AUC (OvR)
        macro_auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")

        # Plot a simple macro summary: average curve is not uniquely defined,
        # so we show macro-AUC and plot all class ROC curves lightly.
        ax = axes[2]
        for i, cls in enumerate(class_names):
            ax.plot(fpr[i], tpr[i], alpha=0.6, label=str(cls))
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{title_prefix} — ROC (OvR), macro-AUC={macro_auc:.3f}")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.show()

def evaluate_multiclass(y_true, y_pred, y_proba=None, model_name="Model"):
    """
    Prints standard multiclass metrics. If y_proba provided, also prints log loss and macro ROC-AUC.
    """
    print(f"\n===== {model_name} =====")
    print(classification_report(y_true, y_pred))
    print("Balanced Accuracy:", balanced_accuracy_score(y_true, y_pred))
    print("Macro F1:", f1_score(y_true, y_pred, average="macro"))

    if y_proba is not None:
        print("Log Loss:", log_loss(y_true, y_proba))
        print("Macro ROC-AUC (OvR):", roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))