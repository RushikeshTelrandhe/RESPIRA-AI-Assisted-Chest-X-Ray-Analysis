from pathlib import Path

# Run this script from the Respira project root.
# Example:
# PS C:\Users\rushi\Desktop\Respira> python setup_project.py

PROJECT_ROOT = Path(__file__).resolve().parent

# Original dataset is kept untouched at:
# PROJECT_ROOT / "Lung_Disease_Dataset"

PREPROCESSING = PROJECT_ROOT / "preprocessing"
DATA = PROJECT_ROOT / "data"

# Preprocessing workspace
preprocessing_dirs = [
    PREPROCESSING,
    PREPROCESSING / "logs",
    PREPROCESSING / "reports",
]

# Processed-data workspace
data_dirs = [
    DATA,
    DATA / "acquired",
    DATA / "cleaned",
    DATA / "balanced",
    DATA / "clahe",
    DATA / "masked",
    DATA / "processed",
    DATA / "processed" / "train",
    DATA / "processed" / "val",
    DATA / "processed" / "test",
    DATA / "manifests",
    DATA / "reports",
]

for directory in preprocessing_dirs + data_dirs:
    directory.mkdir(parents=True, exist_ok=True)

print("\nProject folders created successfully.\n")

print("Preprocessing:")
for directory in preprocessing_dirs:
    print(f"  {directory.relative_to(PROJECT_ROOT)}")

print("\nData:")
for directory in data_dirs:
    print(f"  {directory.relative_to(PROJECT_ROOT)}")

print("\nOriginal dataset is NOT modified:")
print(f"  {PROJECT_ROOT / 'Lung_Disease_Dataset'}")

print("\nFinal structure:")
print("""
Respira/
│
├── env/
│
├── Lung_Disease_Dataset/       # ORIGINAL DATASET - READ ONLY
│   ├── train/
│   ├── val/
│   └── test/
│
├── preprocessing/              # ALL PREPROCESSING SCRIPTS
│   ├── logs/
│   └── reports/
│
└── data/                       # ALL GENERATED DATA
    ├── acquired/
    ├── cleaned/
    ├── balanced/
    ├── clahe/
    ├── masked/
    ├── processed/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── manifests/
    └── reports/
""")
