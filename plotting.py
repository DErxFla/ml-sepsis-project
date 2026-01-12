# =========================
# plots.py  (general utils)
# =========================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

def plot_counts_ax(DEFAULT_BLUE_PALETTE, series: pd.Series, title: str, ax, rotate_xticks: int = 45, order=None, palette=None, missing_label="Missing"):
    sns.set(style="whitegrid", context="talk")

    s = series.astype("object").where(~series.isna(), missing_label)

    vc = s.value_counts(dropna=False).rename("count").reset_index()
    vc.columns = ["category", "count"]

    if order is None:
        # keep most frequent first (except Missing last if present)
        order = vc["category"].tolist()
        if missing_label in order:
            order = [x for x in order if x != missing_label] + [missing_label]

    # --- default blue hues if no palette provided ---
    if palette is None:
        n = len(order)
        colors = sns.color_palette(DEFAULT_BLUE_PALETTE, n_colors=n)
        palette = {cat: colors[i] for i, cat in enumerate(order)}

        # make missing subtle if present
        if missing_label in palette:
            palette[missing_label] = "#c7d7ea"

    sns.barplot(
        data=vc,
        x="category",
        y="count",
        hue="category",
        order=order,
        palette=palette,
        legend=False,
        ax=ax
    )

    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=rotate_xticks)

def plot_mars_counts_ax(MARS_ORDER, MARS_PALETTE, series: pd.Series, title: str, ax, missing_label="Missing"):
    sns.set(style="whitegrid", context="talk")

    s = series.astype("object").where(~series.isna(), missing_label)

    vc = s.value_counts(dropna=False).rename("count").reset_index()
    vc.columns = ["endotype", "count"]

    order = list(MARS_ORDER) + ([missing_label] if missing_label in vc["endotype"].values else [])
    palette = {**MARS_PALETTE, missing_label: "#c7d7ea"}  # subtle blue-ish missing

    sns.barplot(
        data=vc,
        x="endotype",
        y="count",
        hue="endotype",
        order=order,
        palette=palette,
        legend=False,
        ax=ax
    )

    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)

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

def plot_expression_distributions(
    X: pd.DataFrame,
    title_prefix="Expression",
    color="#1f77b4"   # default blue
):
    """
    Overall distribution and per-sample medians.
    X: samples x genes
    """
    sns.set(style="whitegrid", context="talk")

    # Flatten expression values
    vals = X.values.ravel()
    vals = vals[~np.isnan(vals)]

    # Per-sample median expression
    med = X.median(axis=1, skipna=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    # --- All values distribution ---
    sns.histplot(
        vals,
        bins=60,
        kde=True,
        color=color,
        edgecolor="white",
        ax=axes[0]
    )
    axes[0].set_title(f"{title_prefix}: all values")
    axes[0].set_xlabel("Expression")
    axes[0].set_ylabel("Count")

    # --- Per-sample median distribution ---
    sns.histplot(
        med,
        bins=40,
        kde=True,
        color=color,
        edgecolor="white",
        ax=axes[1]
    )
    axes[1].set_title(f"{title_prefix}: per-sample median")
    axes[1].set_xlabel("Median expression")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.show()

    return med

def plot_mars_mortality_summary(
    endotype: pd.Series,
    mortality: pd.Series,
    MARS_ORDER,
    MARS_PALETTE,
    figsize=(12, 4),
    title_suffix=""
):
    """
    Plot 28-day mortality rate and sample counts by Mars endotype.

    endotype: Series of Mars labels (Mars1–Mars4)
    mortality: Series of 0/1 mortality labels
    """

    sns.set(style="whitegrid", context="talk")

    # --- Align & clean ---
    tmp = pd.DataFrame(
        {"endotype": endotype, "mortality28": mortality}
    ).dropna()

    # Ensure categorical order
    tmp["endotype"] = pd.Categorical(
        tmp["endotype"],
        categories=MARS_ORDER,
        ordered=True
    )

    # --- Aggregate ---
    mort_rate = (
        tmp.groupby("endotype", observed=True)["mortality28"]
        .mean()
        .reset_index(name="mortality_rate")
    )

    counts = (
        tmp["endotype"]
        .value_counts()
        .reindex(MARS_ORDER)
        .reset_index()
    )
    counts.columns = ["endotype", "count"]

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    # --- Left: mortality rate ---
    sns.barplot(
        data=mort_rate,
        x="endotype",
        y="mortality_rate",
        hue="endotype",
        palette=MARS_PALETTE,
        legend=False,
        ax=axes[0]
    )
    axes[0].set_title(f"28-day mortality rate by Mars endotype{title_suffix}")
    axes[0].set_ylabel("Mortality rate")
    axes[0].set_xlabel("")
    axes[0].set_ylim(0, 1)

    # --- Right: sample counts ---
    sns.barplot(
        data=counts,
        x="endotype",
        y="count",
        hue="endotype",
        palette=MARS_PALETTE,
        legend=False,
        ax=axes[1]
    )
    axes[1].set_title("Sample counts per Mars endotype\n(where mortality known)")
    axes[1].set_ylabel("Count")
    axes[1].set_xlabel("")

    plt.show()

    return mort_rate, counts

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
    debug: bool = True,
    # ---- styling controls ----
    is_mars: bool = False,
    MARS_ORDER=None,
    MARS_PALETTE=None,
    palette=None,                 # optional override for non-mars
    missing_label: str = "Missing",
    show_legend: bool = True,
):
    """
    PCA plot on an existing axis.
    - If is_mars=True, uses MARS_ORDER + MARS_PALETTE for consistent coloring.
    - Otherwise uses blue hues by default (unless palette is provided).
    Returns (pc_df, pca) or (None, None) if skipped.
    """
    sns.set(style="whitegrid", context="talk")

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

    # --- Guardrail ---
    if len(idx) < min_samples:
        if debug:
            print(f"[plot_pca_ax] SKIP: need >= {min_samples} samples, got {len(idx)}.")
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
    exp = pca.explained_variance_ratio_ * 100

    pc_df = pd.DataFrame(pcs, index=X_use.index, columns=["PC1", "PC2"])

    # --- Labels for plotting ---
    lab_plot = lab.astype("object").where(~lab.isna(), missing_label)

    plot_df = pc_df.copy()
    plot_df["label"] = lab_plot.values

    # --- Palette logic ---
    if is_mars:
        if MARS_ORDER is None or MARS_PALETTE is None:
            raise ValueError("For is_mars=True, you must pass MARS_ORDER and MARS_PALETTE.")
        hue_order = list(MARS_ORDER) + ([missing_label] if (lab_plot == missing_label).any() else [])
        pal = {**MARS_PALETTE, missing_label: "#c7d7ea"}  # subtle blue-ish missing
    else:
        # Default: blue hues for everything non-mars, unless user provides palette
        if palette is None:
            cats = list(pd.unique(plot_df["label"]))
            # keep Missing last if present
            if missing_label in cats:
                cats = [c for c in cats if c != missing_label] + [missing_label]
            hue_order = cats
            colors = sns.color_palette("Blues", n_colors=len(hue_order))
            pal = {c: colors[i] for i, c in enumerate(hue_order)}
            if missing_label in pal:
                pal[missing_label] = "#c7d7ea"
        else:
            pal = palette
            hue_order = None

    # --- Plot ---
    sns.scatterplot(
        data=plot_df,
        x="PC1",
        y="PC2",
        hue="label",
        hue_order=hue_order,
        palette=pal,
        s=point_size,
        alpha=alpha,
        ax=ax,
        edgecolor=None
    )

    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({exp[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({exp[1]:.1f}%)")

    if show_legend:
        ax.legend(title="", fontsize=12, loc="best", frameon=True)
    else:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    return pc_df, pca

def plot_cm_and_optional_roc(
    y_true,
    y_pred,
    y_proba,
    class_names,
    title_prefix="Model",
    # --- styling ---
    cm_cmap="Blues",                 # keep matrices blue
    roc_palette=None,                # pass MARS_PALETTE here for Mars ROC colors
    cm_annot_fontsize=12,
    tick_fontsize=11,
    title_fontsize=13,
    rotate_xticks=45,
    rotate_yticks=0,
    show_colorbar=False
):
    """
    Plots:
      1) Confusion matrix (counts)  - readable heatmap (Blues)
      2) Confusion matrix (normalized by true class) - readable heatmap (Blues)
      3) ROC (OvR), macro-AUC if probabilities provided; ROC lines can use roc_palette.
    """

    sns.set(style="whitegrid", context="talk")

    ncols = 3 if y_proba is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6.5 * ncols, 5), constrained_layout=True)

    # -------------------------
    # Confusion matrix: counts
    # -------------------------
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    ax = axes[0]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cm_cmap,
        cbar=show_colorbar,
        square=True,
        linewidths=0.5,
        linecolor="white",
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": cm_annot_fontsize},
        ax=ax
    )
    ax.set_title(f"{title_prefix} — Confusion Matrix (Counts)", fontsize=title_fontsize)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.tick_params(axis="x", labelrotation=rotate_xticks, labelsize=tick_fontsize)
    ax.tick_params(axis="y", labelrotation=rotate_yticks, labelsize=tick_fontsize)

    # -----------------------------------------
    # Confusion matrix: normalized (row-wise)
    # -----------------------------------------
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax = axes[1]
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap=cm_cmap,
        vmin=0,
        vmax=1,
        cbar=show_colorbar,
        square=True,
        linewidths=0.5,
        linecolor="white",
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": cm_annot_fontsize},
        ax=ax
    )
    ax.set_title(f"{title_prefix} — Confusion Matrix (Normalized)", fontsize=title_fontsize)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.tick_params(axis="x", labelrotation=rotate_xticks, labelsize=tick_fontsize)
    ax.tick_params(axis="y", labelrotation=rotate_yticks, labelsize=tick_fontsize)

    # -------------------------
    # ROC (OvR), macro-AUC
    # -------------------------
    if y_proba is not None:
        ax = axes[2]

        # Binarize true labels for OvR ROC
        y_true_bin = label_binarize(y_true, classes=class_names)

        # Choose ROC colors:
        # - If roc_palette is provided (dict like MARS_PALETTE), use it
        # - Else default blue hues
        if roc_palette is None:
            roc_colors = sns.color_palette("Blues", n_colors=len(class_names) + 2)[2:]
            color_for = {cls: roc_colors[i] for i, cls in enumerate(class_names)}
        else:
            color_for = {cls: roc_palette.get(cls, "#1f77b4") for cls in class_names}

        for i, cls in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            ax.plot(
                fpr, tpr,
                linewidth=2.5,
                alpha=0.95,
                color=color_for[cls],
                label=str(cls)
            )

        # Chance line
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)

        macro_auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{title_prefix} — ROC (OvR)\nMacro-AUC = {macro_auc:.3f}", fontsize=title_fontsize)
        ax.legend(title="Class", fontsize=10, frameon=True, loc="lower right")

    plt.show()

def plot_pca_true_vs_pred(
    X: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
    title: str,
    ax=None,
    scale: bool = True,
    impute: bool = True,
    dropna_true: bool = True,
    min_samples: int = 3,
    alpha: float = 0.85,
    point_size: int = 55,

    # --- misclassification marker controls ---
    mark_misclassified: bool = True,
    mis_marker: str = "x",
    mis_alpha: float = 0.75,
    mis_lw: float = 1.5,          
    mis_size_mult: float = 1.4,   

    # --- palette control ---
    is_mars: bool = False,
    MARS_ORDER=None,
    MARS_PALETTE=None,
    palette=None,
    missing_label: str = "Missing",
    debug: bool = False,
):
    sns.set(style="whitegrid", context="talk")

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(9, 5), constrained_layout=True)

    # Align
    y_true = y_true.reindex(X.index)
    y_pred = y_pred.reindex(X.index)

    idx = y_true.dropna().index if dropna_true else X.index
    if len(idx) < min_samples:
        ax.set_title(title + " (SKIPPED)")
        ax.axis("off")
        return None, None

    X_use = X.loc[idx]
    yt = y_true.loc[idx].astype("object").where(~y_true.loc[idx].isna(), missing_label)
    yp = y_pred.loc[idx].astype("object").where(~y_pred.loc[idx].isna(), missing_label)

    # Preprocess
    if impute:
        X_vals = SimpleImputer(strategy="median").fit_transform(X_use)
    else:
        X_vals = X_use.values

    if scale:
        X_vals = StandardScaler().fit_transform(X_vals)

    # PCA
    pca = PCA(n_components=2, random_state=0)
    pcs = pca.fit_transform(X_vals)
    exp = pca.explained_variance_ratio_ * 100

    plot_df = pd.DataFrame(pcs, index=X_use.index, columns=["PC1", "PC2"])
    plot_df["true"] = yt.values
    plot_df["pred"] = yp.values
    plot_df["correct"] = (plot_df["true"] == plot_df["pred"])

    # Palette for TRUE labels
    if is_mars:
        hue_order = list(MARS_ORDER)
        pal = {**MARS_PALETTE}
    else:
        hue_order = plot_df["true"].unique().tolist()
        colors = sns.color_palette("Blues", n_colors=len(hue_order))
        pal = dict(zip(hue_order, colors))

    # Marker shapes for predicted labels
    pred_levels = plot_df["pred"].unique().tolist()
    marker_pool = ["o", "s", "D", "^", "v", "P", "X"]
    markers = dict(zip(pred_levels, marker_pool[:len(pred_levels)]))

    sns.scatterplot(
        data=plot_df,
        x="PC1",
        y="PC2",
        hue="true",
        hue_order=hue_order,
        palette=pal,
        style="pred",
        markers=markers,
        s=point_size,
        alpha=alpha,
        ax=ax,
        edgecolor=None
    )

    # --- thin cross for misclassified ---
    if mark_misclassified:
        wrong = plot_df.loc[~plot_df["correct"]]
        if len(wrong) > 0:
            ax.scatter(
                wrong["PC1"],
                wrong["PC2"],
                marker=mis_marker,
                s=point_size * mis_size_mult,
                linewidths=mis_lw,
                color="black",
                alpha=mis_alpha,
                zorder=5,
                label="Misclassified"
            )

    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({exp[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({exp[1]:.1f}%)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

    return plot_df, pca


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