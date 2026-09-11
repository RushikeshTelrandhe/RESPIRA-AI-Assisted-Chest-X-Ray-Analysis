"""
Respira - DenseNet121 Experiment Setup

Creates the professional experiment structure required for
DenseNet121 feature extraction and subsequent classification.

Run from project root:

    python scripts/setup_densenet_experiment.py
"""

from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# DIRECTORIES
# ============================================================

DIRECTORIES = [

    # --------------------------------------------------------
    # Source code
    # --------------------------------------------------------

    PROJECT_ROOT / "src",
    PROJECT_ROOT / "src" / "datasets",
    PROJECT_ROOT / "src" / "models",
    PROJECT_ROOT / "src" / "training",
    PROJECT_ROOT / "src" / "evaluation",
    PROJECT_ROOT / "src" / "explainability",
    PROJECT_ROOT / "src" / "utils",

    # --------------------------------------------------------
    # DenseNet121 experiment
    # --------------------------------------------------------

    PROJECT_ROOT / "experiments" / "densenet121",
    PROJECT_ROOT / "experiments" / "densenet121" / "checkpoints",
    PROJECT_ROOT / "experiments" / "densenet121" / "features",
    PROJECT_ROOT / "experiments" / "densenet121" / "logs",
    PROJECT_ROOT / "experiments" / "densenet121" / "metrics",
    PROJECT_ROOT / "experiments" / "densenet121" / "plots",
    PROJECT_ROOT / "experiments" / "densenet121" / "predictions",

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    PROJECT_ROOT / "configs",

    # --------------------------------------------------------
    # Notebooks
    # --------------------------------------------------------

    PROJECT_ROOT / "notebooks",

    # --------------------------------------------------------
    # Training / utility scripts
    # --------------------------------------------------------

    PROJECT_ROOT / "scripts",

    # --------------------------------------------------------
    # Deployment
    # --------------------------------------------------------

    PROJECT_ROOT / "deployment",

    # --------------------------------------------------------
    # Tests
    # --------------------------------------------------------

    PROJECT_ROOT / "tests",
]


# ============================================================
# PYTHON PACKAGE FILES
# ============================================================

INIT_FILES = [

    PROJECT_ROOT / "src" / "__init__.py",

    PROJECT_ROOT / "src" / "datasets" / "__init__.py",
    PROJECT_ROOT / "src" / "models" / "__init__.py",
    PROJECT_ROOT / "src" / "training" / "__init__.py",
    PROJECT_ROOT / "src" / "evaluation" / "__init__.py",
    PROJECT_ROOT / "src" / "explainability" / "__init__.py",
    PROJECT_ROOT / "src" / "utils" / "__init__.py",
]


# ============================================================
# CREATE DIRECTORIES
# ============================================================

print("=" * 70)
print("RESPIRA - DENSENET121 EXPERIMENT SETUP")
print("=" * 70)

print("\nProject root:")
print(PROJECT_ROOT)

print("\nCreating directories...\n")

for directory in DIRECTORIES:

    directory.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"✓ {directory.relative_to(PROJECT_ROOT)}")


# ============================================================
# CREATE __init__.py FILES
# ============================================================

print("\nCreating Python package files...\n")

for init_file in INIT_FILES:

    if not init_file.exists():

        init_file.write_text(
            "",
            encoding="utf-8"
        )

        print(
            f"✓ Created "
            f"{init_file.relative_to(PROJECT_ROOT)}"
        )

    else:

        print(
            f"✓ Exists  "
            f"{init_file.relative_to(PROJECT_ROOT)}"
        )


# ============================================================
# VERIFY IMPORTANT DATA DIRECTORIES
# ============================================================

print("\n" + "=" * 70)
print("DATA DIRECTORY CHECK")
print("=" * 70)

DATA_ROOT = PROJECT_ROOT / "data"

REQUIRED_DATA_DIRECTORIES = [
    DATA_ROOT / "processed",
    DATA_ROOT / "manifests",
]

for directory in REQUIRED_DATA_DIRECTORIES:

    if directory.exists():

        print(
            f"✓ {directory.relative_to(PROJECT_ROOT)}"
        )

    else:

        print(
            f"⚠ Missing: "
            f"{directory.relative_to(PROJECT_ROOT)}"
        )


# ============================================================
# FINAL STRUCTURE
# ============================================================

print("\n" + "=" * 70)
print("DENSENET121 EXPERIMENT STRUCTURE")
print("=" * 70)

DENSENET_ROOT = (
    PROJECT_ROOT
    / "experiments"
    / "densenet121"
)

for directory in sorted(
    DENSENET_ROOT.rglob("*")
):

    if directory.is_dir():

        print(
            f"  {directory.relative_to(PROJECT_ROOT)}"
        )


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("✓ DENSENET121 PROJECT STRUCTURE READY")
print("=" * 70)

print(
    "\nNext stage:"
    "\n  1. Dataset loader"
    "\n  2. DenseNet121 model"
    "\n  3. Feature extraction"
    "\n  4. Feature verification"
)