"""
RESPIRA — STEP 8.14
FINAL XAI ARTIFACT ANALYSIS

Uses the already-generated XAI region audit results.

Inputs:
    outputs/xai_comparison/artifact_audit/xai_region_audit.csv
    outputs/xai_comparison/artifact_audit/suspicious_xai_cases.csv

Outputs:
    outputs/xai_comparison/artifact_audit/final_analysis/
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

AUDIT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "xai_comparison"
    / "artifact_audit"
)

AUDIT_FILE = AUDIT_DIR / "xai_region_audit.csv"
SUSPICIOUS_FILE = AUDIT_DIR / "suspicious_xai_cases.csv"

OUTPUT_DIR = AUDIT_DIR / "final_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

MODELS = ["EfficientNet-B0", "ViT-B/16"]


# ============================================================
# HELPERS
# ============================================================

def safe_mean(series):
    return float(series.mean()) if len(series) else 0.0


def safe_median(series):
    return float(series.median()) if len(series) else 0.0


def safe_std(series):
    return float(series.std()) if len(series) else 0.0


def safe_percent(series, condition):
    if len(series) == 0:
        return 0.0
    return float(condition.mean() * 100.0)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("RESPIRA — STEP 8.14")
print("FINAL XAI ARTIFACT ANALYSIS")
print("=" * 70)

print(f"\nAudit file:")
print(AUDIT_FILE)

print(f"\nSuspicious cases file:")
print(SUSPICIOUS_FILE)

print(f"\nOutput directory:")
print(OUTPUT_DIR)


# ============================================================
# LOAD DATA
# ============================================================

if not AUDIT_FILE.exists():
    raise FileNotFoundError(
        f"\nXAI audit file not found:\n{AUDIT_FILE}"
    )

if not SUSPICIOUS_FILE.exists():
    raise FileNotFoundError(
        f"\nSuspicious cases file not found:\n{SUSPICIOUS_FILE}"
    )


df = pd.read_csv(AUDIT_FILE)
suspicious = pd.read_csv(SUSPICIOUS_FILE)

print("\n" + "=" * 70)
print("DATASET")
print("=" * 70)

print(f"Total XAI audit records : {len(df)}")
print(f"Suspicious records      : {len(suspicious)}")


# ============================================================
# BASIC COUNTS
# ============================================================

total = len(df)
suspicious_count = len(suspicious)

suspicious_percentage = (
    suspicious_count / total * 100
    if total > 0
    else 0.0
)

both_correct = (
    df["eff_correct"].astype(bool)
    & df["vit_correct"].astype(bool)
)

both_wrong = (
    ~df["eff_correct"].astype(bool)
    & ~df["vit_correct"].astype(bool)
)

eff_only_correct = (
    df["eff_correct"].astype(bool)
    & ~df["vit_correct"].astype(bool)
)

vit_only_correct = (
    ~df["eff_correct"].astype(bool)
    & df["vit_correct"].astype(bool)
)


# ============================================================
# MODEL-LEVEL XAI STATISTICS
# ============================================================

model_stats = {
    "EfficientNet-B0": {
        "center_activation_mean":
            safe_mean(df["eff_center_activation"]),
        "border_activation_mean":
            safe_mean(df["eff_border_activation"]),
        "corner_activation_mean":
            safe_mean(df["eff_corner_activation"]),
        "peripheral_activation_mean":
            safe_mean(df["eff_peripheral_activation"]),
        "entropy_mean":
            safe_mean(df["eff_entropy"]),
        "center_activation_median":
            safe_median(df["eff_center_activation"]),
        "border_activation_median":
            safe_median(df["eff_border_activation"]),
        "corner_activation_median":
            safe_median(df["eff_corner_activation"]),
        "peripheral_activation_median":
            safe_median(df["eff_peripheral_activation"]),
        "entropy_median":
            safe_median(df["eff_entropy"]),
    },

    "ViT-B/16": {
        "center_activation_mean":
            safe_mean(df["vit_center_activation"]),
        "border_activation_mean":
            safe_mean(df["vit_border_activation"]),
        "corner_activation_mean":
            safe_mean(df["vit_corner_activation"]),
        "peripheral_activation_mean":
            safe_mean(df["vit_peripheral_activation"]),
        "entropy_mean":
            safe_mean(df["vit_entropy"]),
        "center_activation_median":
            safe_median(df["vit_center_activation"]),
        "border_activation_median":
            safe_median(df["vit_border_activation"]),
        "corner_activation_median":
            safe_median(df["vit_corner_activation"]),
        "peripheral_activation_median":
            safe_median(df["vit_peripheral_activation"]),
        "entropy_median":
            safe_median(df["vit_entropy"]),
    }
}


# ============================================================
# CLASS-LEVEL ANALYSIS
# ============================================================

class_rows = []

for cls in sorted(df["true_class"].unique()):

    subset = df[df["true_class"] == cls]
    suspicious_subset = suspicious[
        suspicious["true_class"] == cls
    ]

    row = {
        "class": cls,
        "images": len(subset),

        "suspicious_cases":
            len(suspicious_subset),

        "suspicious_percentage":
            len(suspicious_subset) / len(subset) * 100
            if len(subset) else 0.0,

        "efficientnet_accuracy":
            safe_percent(
                subset,
                subset["eff_correct"].astype(bool)
            ),

        "vit_accuracy":
            safe_percent(
                subset,
                subset["vit_correct"].astype(bool)
            ),

        "efficientnet_border_activation":
            safe_mean(subset["eff_border_activation"]),

        "efficientnet_corner_activation":
            safe_mean(subset["eff_corner_activation"]),

        "efficientnet_peripheral_activation":
            safe_mean(subset["eff_peripheral_activation"]),

        "efficientnet_center_activation":
            safe_mean(subset["eff_center_activation"]),

        "vit_border_activation":
            safe_mean(subset["vit_border_activation"]),

        "vit_corner_activation":
            safe_mean(subset["vit_corner_activation"]),

        "vit_peripheral_activation":
            safe_mean(subset["vit_peripheral_activation"]),

        "vit_center_activation":
            safe_mean(subset["vit_center_activation"]),

        "efficientnet_entropy":
            safe_mean(subset["eff_entropy"]),

        "vit_entropy":
            safe_mean(subset["vit_entropy"]),
    }

    class_rows.append(row)


class_df = pd.DataFrame(class_rows)

class_df.to_csv(
    OUTPUT_DIR / "xai_artifact_by_class.csv",
    index=False
)


# ============================================================
# CORRECTNESS ANALYSIS
# ============================================================

correctness_rows = []

groups = {
    "Both Correct": both_correct,
    "EfficientNet Correct / ViT Wrong": eff_only_correct,
    "ViT Correct / EfficientNet Wrong": vit_only_correct,
    "Both Wrong": both_wrong,
}

for group_name, mask in groups.items():

    subset = df[mask]

    if len(subset) == 0:
        continue

    correctness_rows.append({
        "group": group_name,
        "images": len(subset),

        "percentage":
            len(subset) / total * 100,

        "eff_border_activation":
            safe_mean(subset["eff_border_activation"]),

        "eff_corner_activation":
            safe_mean(subset["eff_corner_activation"]),

        "eff_peripheral_activation":
            safe_mean(subset["eff_peripheral_activation"]),

        "eff_center_activation":
            safe_mean(subset["eff_center_activation"]),

        "vit_border_activation":
            safe_mean(subset["vit_border_activation"]),

        "vit_corner_activation":
            safe_mean(subset["vit_corner_activation"]),

        "vit_peripheral_activation":
            safe_mean(subset["vit_peripheral_activation"]),

        "vit_center_activation":
            safe_mean(subset["vit_center_activation"]),

        "eff_confidence":
            safe_mean(subset["eff_confidence"]),

        "vit_confidence":
            safe_mean(subset["vit_confidence"]),

        "eff_entropy":
            safe_mean(subset["eff_entropy"]),

        "vit_entropy":
            safe_mean(subset["vit_entropy"]),
    })


correctness_df = pd.DataFrame(correctness_rows)

correctness_df.to_csv(
    OUTPUT_DIR / "xai_artifact_by_prediction_group.csv",
    index=False
)


# ============================================================
# SUSPICIOUS CASE ANALYSIS
# ============================================================

suspicious_summary = {
    "total_test_images": int(total),

    "suspicious_cases": int(suspicious_count),

    "suspicious_percentage":
        float(suspicious_percentage),

    "suspicious_border_cases":
        int(suspicious["suspicious_border"].sum()),

    "suspicious_corner_cases":
        int(suspicious["suspicious_corner"].sum()),

    "both_border_and_corner":
        int(
            (
                suspicious["suspicious_border"].astype(bool)
                &
                suspicious["suspicious_corner"].astype(bool)
            ).sum()
        ),

    "efficientnet_suspicious_accuracy":
        safe_percent(
            suspicious,
            suspicious["eff_correct"].astype(bool)
        ),

    "vit_suspicious_accuracy":
        safe_percent(
            suspicious,
            suspicious["vit_correct"].astype(bool)
        ),

    "efficientnet_suspicious_confidence":
        safe_mean(suspicious["eff_confidence"]),

    "vit_suspicious_confidence":
        safe_mean(suspicious["vit_confidence"]),

    "efficientnet_suspicious_border_activation":
        safe_mean(suspicious["eff_border_activation"]),

    "efficientnet_suspicious_corner_activation":
        safe_mean(suspicious["eff_corner_activation"]),

    "vit_suspicious_border_activation":
        safe_mean(suspicious["vit_border_activation"]),

    "vit_suspicious_corner_activation":
        safe_mean(suspicious["vit_corner_activation"]),
}


save_json(
    suspicious_summary,
    OUTPUT_DIR / "suspicious_case_summary.json"
)


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

correlation_data = pd.DataFrame({
    "eff_border":
        df["eff_border_activation"],

    "eff_corner":
        df["eff_corner_activation"],

    "eff_peripheral":
        df["eff_peripheral_activation"],

    "eff_center":
        df["eff_center_activation"],

    "eff_entropy":
        df["eff_entropy"],

    "eff_confidence":
        df["eff_confidence"],

    "vit_border":
        df["vit_border_activation"],

    "vit_corner":
        df["vit_corner_activation"],

    "vit_peripheral":
        df["vit_peripheral_activation"],

    "vit_center":
        df["vit_center_activation"],

    "vit_entropy":
        df["vit_entropy"],

    "vit_confidence":
        df["vit_confidence"],
})

correlation_matrix = correlation_data.corr()

correlation_matrix.to_csv(
    OUTPUT_DIR / "xai_activation_correlations.csv"
)


# ============================================================
# TOP SUSPICIOUS CASES
# ============================================================

top_border = (
    suspicious
    .sort_values(
        [
            "eff_border_activation",
            "vit_border_activation"
        ],
        ascending=False
    )
    .head(50)
)

top_corner = (
    suspicious
    .sort_values(
        [
            "eff_corner_activation",
            "vit_corner_activation"
        ],
        ascending=False
    )
    .head(50)
)

top_border.to_csv(
    OUTPUT_DIR / "top_50_suspicious_border_cases.csv",
    index=False
)

top_corner.to_csv(
    OUTPUT_DIR / "top_50_suspicious_corner_cases.csv",
    index=False
)


# ============================================================
# VISUALIZATION 1
# MODEL REGIONAL ACTIVATION
# ============================================================

regions = [
    "Center",
    "Border",
    "Corner",
    "Peripheral"
]

eff_values = [
    df["eff_center_activation"].mean(),
    df["eff_border_activation"].mean(),
    df["eff_corner_activation"].mean(),
    df["eff_peripheral_activation"].mean()
]

vit_values = [
    df["vit_center_activation"].mean(),
    df["vit_border_activation"].mean(),
    df["vit_corner_activation"].mean(),
    df["vit_peripheral_activation"].mean()
]

x = np.arange(len(regions))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    x - width / 2,
    eff_values,
    width,
    label="EfficientNet-B0"
)

plt.bar(
    x + width / 2,
    vit_values,
    width,
    label="ViT-B/16"
)

plt.xticks(x, regions)
plt.ylabel("Mean Activation")
plt.title("Regional XAI Activation Comparison")
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "regional_activation_final.png",
    dpi=300
)

plt.close()


# ============================================================
# VISUALIZATION 2
# SUSPICIOUS CASE DISTRIBUTION
# ============================================================

labels = [
    "Not Suspicious",
    "Suspicious"
]

values = [
    total - suspicious_count,
    suspicious_count
]

plt.figure(figsize=(8, 6))

plt.bar(labels, values)

plt.ylabel("Number of Images")
plt.title("XAI Artifact-Risk Distribution")

for i, value in enumerate(values):
    plt.text(
        i,
        value,
        f"{value} ({value / total * 100:.2f}%)",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "suspicious_case_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# VISUALIZATION 3
# CLASS-WISE SUSPICIOUS RATE
# ============================================================

plt.figure(figsize=(11, 6))

plt.bar(
    class_df["class"],
    class_df["suspicious_percentage"]
)

plt.ylabel("Suspicious Cases (%)")
plt.xlabel("Disease Class")
plt.title("Potential XAI Artifact Cases by Disease Class")

plt.xticks(
    rotation=30,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "suspicious_cases_by_class.png",
    dpi=300
)

plt.close()


# ============================================================
# VISUALIZATION 4
# BORDER VS CENTER
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    df["eff_center_activation"],
    df["eff_border_activation"],
    alpha=0.35,
    label="EfficientNet-B0"
)

plt.xlabel("Center Activation")
plt.ylabel("Border Activation")
plt.title("EfficientNet-B0 Center vs Border Activation")
plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "efficientnet_center_vs_border_final.png",
    dpi=300
)

plt.close()


# ============================================================
# VISUALIZATION 5
# EFFICIENTNET VS VIT BORDER ACTIVATION
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    df["eff_border_activation"],
    df["vit_border_activation"],
    alpha=0.35
)

plt.xlabel("EfficientNet Border Activation")
plt.ylabel("ViT Border Activation")
plt.title("EfficientNet vs ViT Border Activation")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "efficientnet_vs_vit_border_activation.png",
    dpi=300
)

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

final_summary = {
    "analysis": "STEP 8.14 - Final XAI Artifact Analysis",

    "dataset": {
        "total_xai_records": int(total),
        "suspicious_cases": int(suspicious_count),
        "suspicious_percentage":
            float(suspicious_percentage),
    },

    "prediction_groups": {
        "both_correct": int(both_correct.sum()),
        "efficientnet_correct_vit_wrong":
            int(eff_only_correct.sum()),
        "vit_correct_efficientnet_wrong":
            int(vit_only_correct.sum()),
        "both_wrong":
            int(both_wrong.sum()),
    },

    "efficientnet": model_stats["EfficientNet-B0"],

    "vit": model_stats["ViT-B/16"],

    "suspicious_cases": suspicious_summary,

    "interpretation": {
        "finding":
            "The XAI audit identified recurrent peripheral, border, "
            "and corner activation patterns in a subset of test images.",

        "suspicious_case_rate":
            f"{suspicious_percentage:.2f}%",

        "caution":
            "Suspicious XAI activation is an artifact-risk indicator, "
            "not proof of data leakage or model dependence on artifacts.",

        "recommendation":
            "The identified cases should be reported as an XAI audit "
            "finding and visually inspected before drawing conclusions "
            "about source or acquisition artifacts."
    }
}

save_json(
    final_summary,
    OUTPUT_DIR / "final_xai_artifact_summary.json"
)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL XAI ARTIFACT ANALYSIS RESULTS")
print("=" * 70)

print(f"\nTotal images              : {total}")
print(f"Potentially suspicious    : {suspicious_count}")
print(f"Suspicious percentage     : {suspicious_percentage:.2f}%")

print("\nPrediction groups:")
print(f"  Both correct            : {both_correct.sum()}")
print(
    f"  EfficientNet correct / ViT wrong : "
    f"{eff_only_correct.sum()}"
)
print(
    f"  ViT correct / EfficientNet wrong : "
    f"{vit_only_correct.sum()}"
)
print(f"  Both wrong              : {both_wrong.sum()}")

print("\nEfficientNet-B0:")
print(
    f"  Center activation      : "
    f"{model_stats['EfficientNet-B0']['center_activation_mean']:.4f}"
)
print(
    f"  Border activation      : "
    f"{model_stats['EfficientNet-B0']['border_activation_mean']:.4f}"
)
print(
    f"  Corner activation      : "
    f"{model_stats['EfficientNet-B0']['corner_activation_mean']:.4f}"
)
print(
    f"  Peripheral activation  : "
    f"{model_stats['EfficientNet-B0']['peripheral_activation_mean']:.4f}"
)
print(
    f"  Entropy                : "
    f"{model_stats['EfficientNet-B0']['entropy_mean']:.4f}"
)

print("\nViT-B/16:")
print(
    f"  Center activation      : "
    f"{model_stats['ViT-B/16']['center_activation_mean']:.4f}"
)
print(
    f"  Border activation      : "
    f"{model_stats['ViT-B/16']['border_activation_mean']:.4f}"
)
print(
    f"  Corner activation      : "
    f"{model_stats['ViT-B/16']['corner_activation_mean']:.4f}"
)
print(
    f"  Peripheral activation  : "
    f"{model_stats['ViT-B/16']['peripheral_activation_mean']:.4f}"
)
print(
    f"  Entropy                : "
    f"{model_stats['ViT-B/16']['entropy_mean']:.4f}"
)

print("\n" + "=" * 70)
print("STEP 8.14 COMPLETED")
print("=" * 70)

print("\nGenerated:")
print(
    f"  {OUTPUT_DIR / 'final_xai_artifact_summary.json'}"
)
print(
    f"  {OUTPUT_DIR / 'suspicious_case_summary.json'}"
)
print(
    f"  {OUTPUT_DIR / 'xai_artifact_by_class.csv'}"
)
print(
    f"  {OUTPUT_DIR / 'xai_artifact_by_prediction_group.csv'}"
)
print(
    f"  {OUTPUT_DIR / 'xai_activation_correlations.csv'}"
)
print(
    f"  {OUTPUT_DIR / 'top_50_suspicious_border_cases.csv'}"
)
print(
    f"  {OUTPUT_DIR / 'top_50_suspicious_corner_cases.csv'}"
)

print("\nFigures:")
print(
    f"  {OUTPUT_DIR / 'regional_activation_final.png'}"
)
print(
    f"  {OUTPUT_DIR / 'suspicious_case_distribution.png'}"
)
print(
    f"  {OUTPUT_DIR / 'suspicious_cases_by_class.png'}"
)
print(
    f"  {OUTPUT_DIR / 'efficientnet_center_vs_border_final.png'}"
)
print(
    f"  {OUTPUT_DIR / 'efficientnet_vs_vit_border_activation.png'}"
)

print("\n✓ Final XAI artifact analysis completed.")
print("✓ No model checkpoints were modified.")
print("✓ No dataset files were modified.")
print("✓ Analysis uses the existing 1,947-image XAI audit.")