# ============================================================
# RESPIRA — STEP 8.12
# XAI AUDIT VISUALIZATION
# ============================================================

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

AUDIT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "xai_comparison"
    / "artifact_audit"
)

AUDIT_CSV = (
    AUDIT_ROOT
    / "xai_region_audit.csv"
)

OUTPUT_ROOT = (
    AUDIT_ROOT
    / "visualizations"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("RESPIRA — STEP 8.12")
print("XAI AUDIT VISUALIZATION")
print("=" * 70)

print()
print("Input:")
print(AUDIT_CSV)

if not AUDIT_CSV.exists():
    raise FileNotFoundError(
        f"\nAudit file not found:\n{AUDIT_CSV}"
    )

df = pd.read_csv(
    AUDIT_CSV
)

print()
print(
    f"Images: {len(df)}"
)


# ============================================================
# 1. REGIONAL ACTIVATION COMPARISON
# ============================================================

regional_metrics = [
    "center_activation",
    "border_activation",
    "corner_activation",
    "peripheral_activation",
]

eff_means = [
    df[f"eff_{m}"].mean()
    for m in regional_metrics
]

vit_means = [
    df[f"vit_{m}"].mean()
    for m in regional_metrics
]

labels = [
    "Center",
    "Border",
    "Corner",
    "Peripheral",
]

x = range(len(labels))

plt.figure(
    figsize=(10, 6)
)

width = 0.35

plt.bar(
    [i - width / 2 for i in x],
    eff_means,
    width=width,
    label="EfficientNet-B0"
)

plt.bar(
    [i + width / 2 for i in x],
    vit_means,
    width=width,
    label="ViT-B/16"
)

plt.xticks(
    list(x),
    labels
)

plt.ylabel(
    "Mean activation ratio"
)

plt.title(
    "Regional XAI Activation: EfficientNet-B0 vs ViT-B/16"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_ROOT
    / "regional_activation_comparison.png",
    dpi=250,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 2. BORDER ACTIVATION DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.hist(
    df["eff_border_activation"],
    bins=40,
    alpha=0.6,
    label="EfficientNet-B0"
)

plt.hist(
    df["vit_border_activation"],
    bins=40,
    alpha=0.6,
    label="ViT-B/16"
)

plt.xlabel(
    "Border activation ratio"
)

plt.ylabel(
    "Number of images"
)

plt.title(
    "Border Activation Distribution"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_ROOT
    / "border_activation_distribution.png",
    dpi=250,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 3. CORNER ACTIVATION DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.hist(
    df["eff_corner_activation"],
    bins=40,
    alpha=0.6,
    label="EfficientNet-B0"
)

plt.hist(
    df["vit_corner_activation"],
    bins=40,
    alpha=0.6,
    label="ViT-B/16"
)

plt.xlabel(
    "Corner activation ratio"
)

plt.ylabel(
    "Number of images"
)

plt.title(
    "Corner Activation Distribution"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_ROOT
    / "corner_activation_distribution.png",
    dpi=250,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 4. ENTROPY DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.hist(
    df["eff_entropy"],
    bins=40,
    alpha=0.6,
    label="EfficientNet-B0"
)

plt.hist(
    df["vit_entropy"],
    bins=40,
    alpha=0.6,
    label="ViT-B/16"
)

plt.xlabel(
    "Normalized activation entropy"
)

plt.ylabel(
    "Number of images"
)

plt.title(
    "XAI Activation Entropy Distribution"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_ROOT
    / "activation_entropy_distribution.png",
    dpi=250,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 5. BORDER VS CENTER
# ============================================================

plt.figure(
    figsize=(8, 7)
)

plt.scatter(
    df["eff_center_activation"],
    df["eff_border_activation"],
    alpha=0.35,
    s=12,
    label="EfficientNet-B0"
)

plt.scatter(
    df["vit_center_activation"],
    df["vit_border_activation"],
    alpha=0.35,
    s=12,
    label="ViT-B/16"
)

plt.xlabel(
    "Center activation"
)

plt.ylabel(
    "Border activation"
)

plt.title(
    "Center vs Border XAI Activation"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_ROOT
    / "center_vs_border_activation.png",
    dpi=250,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 6. PER-CLASS BORDER ACTIVATION
# ============================================================

class_border = (
    df.groupby("true_class")
    .agg(
        efficientnet_border=(
            "eff_border_activation",
            "mean"
        ),
        vit_border=(
            "vit_border_activation",
            "mean"
        ),
    )
)

plt.figure(
    figsize=(11, 6)
)

x = range(len(class_border))

plt.bar(
    [i - width / 2 for i in x],
    class_border["efficientnet_border"],
    width=width,
    label="EfficientNet-B0"
)

plt.bar(
    [i + width / 2 for i in x],
    class_border["vit_border"],
    width=width,
    label="ViT-B/16"
)

plt.xticks(
    list(x),
    class_border.index,
    rotation=25,
    ha="right"
)

plt.ylabel(
    "Mean border activation"
)

plt.title(
    "Border Activation by Disease Class"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_ROOT
    / "border_activation_by_class.png",
    dpi=250,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 7. TOP SUSPICIOUS CASES
# ============================================================

df["combined_border_score"] = (
    df["eff_border_activation"]
    + df["vit_border_activation"]
) / 2

df["combined_corner_score"] = (
    df["eff_corner_activation"]
    + df["vit_corner_activation"]
) / 2

top_border = (
    df.sort_values(
        "combined_border_score",
        ascending=False
    )
    .head(50)
)

top_corner = (
    df.sort_values(
        "combined_corner_score",
        ascending=False
    )
    .head(50)
)

top_border.to_csv(
    OUTPUT_ROOT
    / "top_50_border_cases.csv",
    index=False
)

top_corner.to_csv(
    OUTPUT_ROOT
    / "top_50_corner_cases.csv",
    index=False
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("STEP 8.12 COMPLETED")
print("=" * 70)

print()

print(
    "Generated visualizations:"
)

print(
    "  regional_activation_comparison.png"
)

print(
    "  border_activation_distribution.png"
)

print(
    "  corner_activation_distribution.png"
)

print(
    "  activation_entropy_distribution.png"
)

print(
    "  center_vs_border_activation.png"
)

print(
    "  border_activation_by_class.png"
)

print()

print(
    "Generated inspection lists:"
)

print(
    "  top_50_border_cases.csv"
)

print(
    "  top_50_corner_cases.csv"
)

print()

print(
    "Output directory:"
)

print(
    OUTPUT_ROOT
)

print()

print(
    "Next stage:"
)

print(
    "STEP 8.13 — VISUAL INSPECTION OF TOP ARTIFACT CASES"
)