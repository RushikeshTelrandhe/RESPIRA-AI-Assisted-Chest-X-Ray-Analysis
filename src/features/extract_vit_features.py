# ============================================================
# RESPIRA
# STEP 6.4 — VISION TRANSFORMER FEATURE EXTRACTION
# ============================================================
#
# Purpose:
#   Extract internal ViT representations from the best
#   trained ViT-B/16 checkpoint.
#
# Outputs:
#
# outputs/vit/features/
# ├── train/
# │   ├── token_features.pt
# │   ├── cls_features.pt
# │   ├── pooled_features.pt
# │   ├── labels.npy
# │   └── metadata.csv
# │
# ├── val/
# │   ├── token_features.pt
# │   ├── cls_features.pt
# │   ├── pooled_features.pt
# │   ├── labels.npy
# │   └── metadata.csv
# │
# └── test/
#     ├── token_features.pt
#     ├── cls_features.pt
#     ├── pooled_features.pt
#     ├── labels.npy
#     └── metadata.csv
#
# ViT-B/16 @ 224x224:
#
#   Patch tokens : (N, 196, 768)
#   CLS token    : (N, 768)
#   Pooled       : (N, 768)
#
# These features will later be used by the
# DenseNet-ViT cross-attention fusion architecture.
# ============================================================

from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm


# ============================================================
# 1. PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 2. IMPORT MODEL
# ============================================================

from src.models.vit import VisionTransformerModel


# ============================================================
# 3. CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 224

NUM_CLASSES = 6

CLASS_NAMES = [
    "Atelectasis",
    "Bacterial Pneumonia",
    "Normal",
    "Pulmonary Edema",
    "Tuberculosis",
    "Viral Pneumonia",
]

BATCH_SIZE = 32

NUM_WORKERS = 0

PIN_MEMORY = DEVICE.type == "cuda"


# ============================================================
# 4. PATHS
# ============================================================

PROCESSED_DATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "vit"
    / "checkpoints"
    / "best_model.pth"
)

FEATURE_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "vit"
    / "features"
)

SUMMARY_PATH = (
    FEATURE_ROOT
    / "extraction_summary.json"
)


# ============================================================
# 5. PRINT CONFIGURATION
# ============================================================

print("=" * 70)
print("RESPIRA — VISION TRANSFORMER FEATURE EXTRACTION")
print("=" * 70)

print()
print("Project root:")
print(PROJECT_ROOT)

print()
print("Processed dataset:")
print(PROCESSED_DATA)

print()
print("Checkpoint:")
print(CHECKPOINT_PATH)

print()
print("Feature output:")
print(FEATURE_ROOT)

print()
print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

print()
print("Classes:")
for idx, name in enumerate(CLASS_NAMES):
    print(f"  {idx}: {name}")

print()


# ============================================================
# 6. VALIDATE PATHS
# ============================================================

if not PROCESSED_DATA.exists():
    raise FileNotFoundError(
        f"Processed dataset does not exist:\n"
        f"{PROCESSED_DATA}"
    )

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"ViT checkpoint does not exist:\n"
        f"{CHECKPOINT_PATH}"
    )


# ============================================================
# 7. DATASET
# ============================================================

class ChestXrayDataset(Dataset):

    def __init__(
        self,
        root_dir,
        split,
        class_names,
    ):

        self.root_dir = Path(root_dir)
        self.split = split
        self.class_names = class_names

        self.split_dir = (
            self.root_dir / split
        )

        if not self.split_dir.exists():
            raise FileNotFoundError(
                f"Dataset split does not exist:\n"
                f"{self.split_dir}"
            )

        self.class_to_idx = {
            name: idx
            for idx, name in enumerate(class_names)
        }

        self.samples = []

        self._collect_samples()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found in:\n"
                f"{self.split_dir}"
            )

    def _collect_samples(self):

        for class_name in self.class_names:

            class_dir = (
                self.split_dir / class_name
            )

            if not class_dir.exists():

                raise FileNotFoundError(
                    f"Required class directory missing:\n"
                    f"{class_dir}"
                )

            image_paths = sorted(
                [
                    p
                    for p in class_dir.rglob("*")
                    if p.is_file()
                    and p.suffix.lower()
                    in [".png", ".jpg", ".jpeg"]
                ]
            )

            label = self.class_to_idx[
                class_name
            ]

            for image_path in image_paths:

                self.samples.append(
                    {
                        "path": str(image_path),
                        "label": label,
                        "class_name": class_name,
                    }
                )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        item = self.samples[index]

        image = Image.open(
            item["path"]
        ).convert("RGB")

        image = np.asarray(
            image,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Dataset images are already 224x224.
        # Do not perform additional augmentation.
        # ----------------------------------------------------

        if image.shape[:2] != (
            IMAGE_SIZE,
            IMAGE_SIZE,
        ):

            image = np.array(
                Image.fromarray(
                    image.astype(np.uint8)
                ).resize(
                    (
                        IMAGE_SIZE,
                        IMAGE_SIZE
                    )
                )
            ).astype(np.float32)

        image /= 255.0

        # ----------------------------------------------------
        # ImageNet normalization
        # ----------------------------------------------------

        mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32
        )

        std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32
        )

        image = (
            (image - mean)
            / std
        )

        image = torch.from_numpy(
            image
        ).permute(
            2,
            0,
            1
        ).float()

        return (
            image,
            item["label"],
            item["path"],
            item["class_name"],
        )


# ============================================================
# 8. LOAD VIT MODEL
# ============================================================

print("=" * 70)
print("LOADING VISION TRANSFORMER")
print("=" * 70)

model = VisionTransformerModel(
    num_classes=NUM_CLASSES,
    pretrained=False,
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
)

if not isinstance(checkpoint, dict):
    raise RuntimeError(
        "Unexpected checkpoint format."
    )

if "model_state_dict" not in checkpoint:
    raise KeyError(
        "Checkpoint does not contain "
        "'model_state_dict'."
    )

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()

for parameter in model.parameters():
    parameter.requires_grad = False

print("✓ ViT checkpoint loaded.")
print("✓ ViT frozen.")
print()


# ============================================================
# 9. FEATURE EXTRACTION FUNCTION
# ============================================================

@torch.no_grad()
def extract_features(
    dataset,
    split,
):

    print("=" * 70)
    print(f"EXTRACTING ViT FEATURES — {split.upper()}")
    print("=" * 70)

    print()
    print("Images:", len(dataset))
    print("Batch size:", BATCH_SIZE)
    print()

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    token_features = []
    cls_features = []
    pooled_features = []
    labels = []
    paths = []
    class_names = []

    # --------------------------------------------------------
    # Register hook on the ViT transformer output.
    #
    # The exact internal module is obtained from the model.
    # --------------------------------------------------------

    captured = {}

    def hook_fn(module, inputs, output):

        captured["output"] = output

    hook_handle = None

    # --------------------------------------------------------
    # Find transformer encoder if available.
    # --------------------------------------------------------

    if hasattr(model, "vit"):

        vit_backbone = model.vit

    elif hasattr(model, "backbone"):

        vit_backbone = model.backbone

    elif hasattr(model, "model"):

        vit_backbone = model.model

    else:

        vit_backbone = None

    # --------------------------------------------------------
    # Try to register on encoder.
    # --------------------------------------------------------

    if vit_backbone is not None:

        if hasattr(
            vit_backbone,
            "encoder"
        ):

            hook_handle = (
                vit_backbone
                .encoder
                .register_forward_hook(
                    hook_fn
                )
            )

        elif hasattr(
            vit_backbone,
            "encoder_layer"
        ):

            hook_handle = (
                vit_backbone
                .encoder_layer
                .register_forward_hook(
                    hook_fn
                )
            )

    # --------------------------------------------------------
    # Process batches
    # --------------------------------------------------------

    for (
        images,
        batch_labels,
        batch_paths,
        batch_class_names,
    ) in tqdm(
        loader,
        desc=f"Extracting {split}",
    ):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        output = model(images)

        # ----------------------------------------------------
        # Extract transformer representation.
        # ----------------------------------------------------

        features = captured.get(
            "output",
            None
        )

        # ----------------------------------------------------
        # Some model implementations return the
        # transformer representation directly.
        # ----------------------------------------------------

        if features is None:

            if isinstance(output, dict):

                for key in [
                    "last_hidden_state",
                    "hidden_states",
                    "features",
                    "tokens",
                ]:

                    if key in output:

                        features = output[key]

                        if isinstance(
                            features,
                            (tuple, list)
                        ):

                            features = features[-1]

                        break

            elif torch.is_tensor(output):

                features = output

        if features is None:

            raise RuntimeError(
                "Could not capture ViT token features.\n"
                "Please inspect src/models/vit.py."
            )

        # ----------------------------------------------------
        # Expected:
        #
        # (B, 197, 768)
        # ----------------------------------------------------

        if features.ndim != 3:

            raise RuntimeError(
                "Unexpected ViT feature shape: "
                f"{tuple(features.shape)}"
            )

        # ----------------------------------------------------
        # CLS token
        # ----------------------------------------------------

        cls = features[:, 0, :]

        # ----------------------------------------------------
        # Patch tokens
        #
        # 196 tokens for 224x224 / 16x16 patches
        # ----------------------------------------------------

        patches = features[:, 1:, :]

        # ----------------------------------------------------
        # Global mean pooled representation
        # ----------------------------------------------------

        pooled = patches.mean(
            dim=1
        )

        token_features.append(
            patches.cpu()
        )

        cls_features.append(
            cls.cpu()
        )

        pooled_features.append(
            pooled.cpu()
        )

        labels.append(
            batch_labels.cpu()
        )

        paths.extend(
            list(batch_paths)
        )

        class_names.extend(
            list(batch_class_names)
        )

    if hook_handle is not None:
        hook_handle.remove()

    # ========================================================
    # CONCATENATE
    # ========================================================

    token_features = torch.cat(
        token_features,
        dim=0
    )

    cls_features = torch.cat(
        cls_features,
        dim=0
    )

    pooled_features = torch.cat(
        pooled_features,
        dim=0
    )

    labels = torch.cat(
        labels,
        dim=0
    )

    # ========================================================
    # VERIFY
    # ========================================================

    print()
    print("Feature shapes:")

    print(
        "  Patch tokens:",
        tuple(token_features.shape)
    )

    print(
        "  CLS token   :",
        tuple(cls_features.shape)
    )

    print(
        "  Pooled      :",
        tuple(pooled_features.shape)
    )

    print(
        "  Labels      :",
        tuple(labels.shape)
    )

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    output_dir = (
        FEATURE_ROOT / split
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # SAVE FEATURES
    # ========================================================

    token_path = (
        output_dir
        / "token_features.pt"
    )

    cls_path = (
        output_dir
        / "cls_features.pt"
    )

    pooled_path = (
        output_dir
        / "pooled_features.pt"
    )

    labels_path = (
        output_dir
        / "labels.npy"
    )

    metadata_path = (
        output_dir
        / "metadata.csv"
    )

    torch.save(
        token_features,
        token_path
    )

    torch.save(
        cls_features,
        cls_path
    )

    torch.save(
        pooled_features,
        pooled_path
    )

    np.save(
        labels_path,
        labels.numpy()
    )

    # ========================================================
    # METADATA
    # ========================================================

    metadata = pd.DataFrame(
        {
            "index": np.arange(
                len(paths)
            ),
            "path": paths,
            "label": labels.numpy(),
            "class_name": class_names,
        }
    )

    metadata.to_csv(
        metadata_path,
        index=False
    )

    # ========================================================
    # PRINT OUTPUTS
    # ========================================================

    print()
    print("✓ Saved patch tokens:")
    print(" ", token_path)

    print()
    print("✓ Saved CLS features:")
    print(" ", cls_path)

    print()
    print("✓ Saved pooled features:")
    print(" ", pooled_path)

    print()
    print("✓ Saved labels:")
    print(" ", labels_path)

    print()
    print("✓ Saved metadata:")
    print(" ", metadata_path)

    return {
        "images": len(dataset),
        "token_shape": list(
            token_features.shape
        ),
        "cls_shape": list(
            cls_features.shape
        ),
        "pooled_shape": list(
            pooled_features.shape
        ),
    }


# ============================================================
# 10. RUN EXTRACTION
# ============================================================

summary = {}

for split in [
    "train",
    "val",
    "test",
]:

    dataset = ChestXrayDataset(
        root_dir=PROCESSED_DATA,
        split=split,
        class_names=CLASS_NAMES,
    )

    summary[split] = extract_features(
        dataset,
        split,
    )

    print()


# ============================================================
# 11. SAVE SUMMARY
# ============================================================

FEATURE_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

summary_data = {
    "model": "ViT-B/16",
    "checkpoint": str(
        CHECKPOINT_PATH
    ),
    "image_size": IMAGE_SIZE,
    "num_classes": NUM_CLASSES,
    "classes": CLASS_NAMES,
    "device": str(DEVICE),
    "feature_dimensions": {
        "patch_tokens": 768,
        "num_patch_tokens": 196,
        "cls_features": 768,
        "pooled_features": 768,
    },
    "splits": summary,
}

with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary_data,
        f,
        indent=4
    )


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("=" * 70)
print("VISION TRANSFORMER FEATURE EXTRACTION COMPLETE")
print("=" * 70)

print()
print("Features saved to:")
print(FEATURE_ROOT)

print()
print("Expected representation:")
print("  Patch tokens : (N, 196, 768)")
print("  CLS token    : (N, 768)")
print("  Pooled       : (N, 768)")

print()
print("✓ Train features extracted.")
print("✓ Validation features extracted.")
print("✓ Test features extracted.")
print("✓ Patch tokens saved.")
print("✓ CLS features saved.")
print("✓ Pooled features saved.")
print("✓ Labels saved.")
print("✓ Metadata saved.")
print("✓ Extraction summary saved.")
    
print("=" * 70)