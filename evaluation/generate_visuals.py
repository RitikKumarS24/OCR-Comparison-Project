import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV = os.path.join(PROJECT_ROOT, "results", "raw_outputs", "benchmark_results.csv")
GRAPHS_DIR = os.path.join(PROJECT_ROOT, "results", "graphs")
CONDITION_CSV = os.path.join(PROJECT_ROOT, "results", "summary_by_condition.csv")

ENGINE_ORDER = ["Tesseract", "EasyOCR", "PaddleOCR", "docTR", "SuryaOCR"]
CONDITION_ORDER = ["original", "noisy", "blurry", "rotated", "low_resolution"]


def generate_all_visuals():
    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(f"Benchmark CSV not found at {RAW_CSV}")

    os.makedirs(GRAPHS_DIR, exist_ok=True)
    df = pd.read_csv(RAW_CSV)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    palette = sns.color_palette("Set2", n_colors=5)

    # Save detailed condition breakdown CSV
    cond_summary = (
        df[df["pipeline"] == "preprocessed"]
        .groupby(["engine", "language", "condition"], as_index=False)
        .agg(
            avg_char_acc=("char_accuracy_pct", "mean"),
            avg_word_acc=("word_accuracy_pct", "mean"),
            avg_cer=("cer", "mean"),
            avg_wer=("wer", "mean"),
            avg_latency_sec=("latency_sec", "mean"),
        )
        .round(4)
    )
    cond_summary.to_csv(CONDITION_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved condition breakdown table -> {CONDITION_CSV}")

    # -------------------------------------------------------------------------
    # Chart 1: Character Accuracy (%) by Engine & Language (Preprocessed)
    # -------------------------------------------------------------------------
    prep_df = df[df["pipeline"] == "preprocessed"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(
        data=prep_df,
        x="engine",
        y="char_accuracy_pct",
        hue="language",
        order=ENGINE_ORDER,
        palette=["#2b5c8f", "#d95f02"],
        errorbar=None,
        ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=3, fontsize=9.5, fontweight="bold")
    ax.set_title("Character Recognition Accuracy (%) by OCR Engine & Language (Preprocessed)", fontweight="bold", pad=12)
    ax.set_xlabel("OCR Engine", fontweight="bold")
    ax.set_ylabel("Mean Character Accuracy (%)", fontweight="bold")
    ax.set_ylim(0, 110)
    ax.legend(title="Language", loc="upper right")
    plt.tight_layout()
    c1_path = os.path.join(GRAPHS_DIR, "01_accuracy_by_engine_language.png")
    plt.savefig(c1_path, dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # Chart 2: Impact of OpenCV Preprocessing on Character Error Rate (CER)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=False)
    for idx, lang in enumerate(["English", "Hindi"]):
        sub = df[df["language"] == lang]
        sns.barplot(
            data=sub,
            x="engine",
            y="cer",
            hue="pipeline",
            order=ENGINE_ORDER,
            hue_order=["raw", "preprocessed"],
            palette=["#999999", "#1b9e77"],
            errorbar=None,
            ax=axes[idx],
        )
        for container in axes[idx].containers:
            axes[idx].bar_label(container, fmt="%.3f", padding=2, fontsize=8.5)
        axes[idx].set_title(f"{lang}: Raw vs. Preprocessed CER (Lower is Better)", fontweight="bold")
        axes[idx].set_xlabel("OCR Engine", fontweight="bold")
        axes[idx].set_ylabel("Mean Character Error Rate (CER)", fontweight="bold")
        axes[idx].legend(title="Pipeline")
    plt.tight_layout()
    c2_path = os.path.join(GRAPHS_DIR, "02_raw_vs_preprocessed_impact.png")
    plt.savefig(c2_path, dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # Chart 3: Robustness Heatmap Across 5 Distortion Conditions (Preprocessed)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for idx, lang in enumerate(["English", "Hindi"]):
        piv = (
            prep_df[prep_df["language"] == lang]
            .pivot_table(index="engine", columns="condition", values="char_accuracy_pct", aggfunc="mean")
            .reindex(index=ENGINE_ORDER, columns=CONDITION_ORDER)
        )
        sns.heatmap(
            piv,
            annot=True,
            fmt=".1f",
            cmap="YlGnBu",
            vmin=0,
            vmax=100,
            cbar_kws={"label": "Char Accuracy (%)"},
            ax=axes[idx],
        )
        axes[idx].set_title(f"{lang} Accuracy (%) Across Distortion Conditions", fontweight="bold")
        axes[idx].set_xlabel("Document Condition", fontweight="bold")
        axes[idx].set_ylabel("OCR Engine", fontweight="bold")
    plt.tight_layout()
    c3_path = os.path.join(GRAPHS_DIR, "03_robustness_by_condition_heatmap.png")
    plt.savefig(c3_path, dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # Chart 4: Speed vs. Accuracy Trade-Off Scatter Plot (Preprocessed)
    # -------------------------------------------------------------------------
    tradeoff = (
        prep_df.groupby(["engine", "language"], as_index=False)
        .agg(avg_latency=("latency_sec", "mean"), avg_acc=("char_accuracy_pct", "mean"))
    )
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    markers = {"English": "o", "Hindi": "s"}
    for lang in ["English", "Hindi"]:
        sub = tradeoff[tradeoff["language"] == lang]
        for i, row in sub.iterrows():
            c_idx = ENGINE_ORDER.index(row["engine"])
            ax.scatter(
                row["avg_latency"],
                row["avg_acc"],
                s=160,
                color=palette[c_idx],
                marker=markers[lang],
                edgecolors="black",
                linewidth=1.1,
                label=f"{row['engine']} ({lang})",
            )
            ax.annotate(
                f"{row['engine']} ({lang[:2]})",
                (row["avg_latency"], row["avg_acc"]),
                textcoords="offset points",
                xytext=(6, 5),
                fontsize=9,
                fontweight="bold",
            )
    ax.set_xscale("log")
    ax.set_title("Speed vs. Accuracy Trade-Off (Log Latency vs. Mean Character Accuracy)", fontweight="bold", pad=12)
    ax.set_xlabel("Mean Inference Latency per Image (seconds, log scale) — Faster ←", fontweight="bold")
    ax.set_ylabel("Mean Character Accuracy (%) — More Accurate ↑", fontweight="bold")
    ax.set_ylim(0, 108)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    c4_path = os.path.join(GRAPHS_DIR, "04_speed_vs_accuracy_pareto.png")
    plt.savefig(c4_path, dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # Chart 5: Overall CER vs. WER Comparison (English Preprocessed)
    # -------------------------------------------------------------------------
    eng_prep = prep_df[prep_df["language"] == "English"].melt(
        id_vars=["engine"],
        value_vars=["cer", "wer"],
        var_name="Metric",
        value_name="Error Rate",
    )
    eng_prep["Metric"] = eng_prep["Metric"].str.upper()
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    sns.barplot(
        data=eng_prep,
        x="engine",
        y="Error Rate",
        hue="Metric",
        order=ENGINE_ORDER,
        palette=["#377eb8", "#e41a1c"],
        errorbar=None,
        ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=9)
    ax.set_title("Character Error Rate (CER) vs. Word Error Rate (WER) — English", fontweight="bold", pad=12)
    ax.set_xlabel("OCR Engine", fontweight="bold")
    ax.set_ylabel("Error Rate (Lower is Better)", fontweight="bold")
    plt.tight_layout()
    c5_path = os.path.join(GRAPHS_DIR, "05_cer_wer_comparison.png")
    plt.savefig(c5_path, dpi=300)
    plt.close()

    print("\n================ VISUALS GENERATED ================")
    for p in [c1_path, c2_path, c3_path, c4_path, c5_path]:
        print("Created:", p)


if __name__ == "__main__":
    generate_all_visuals()