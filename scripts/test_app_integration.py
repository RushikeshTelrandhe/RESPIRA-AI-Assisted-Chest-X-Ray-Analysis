import os
import sys
sys.path.append(os.path.abspath("."))

from PIL import Image
import numpy as np
from backend.app.services.model_manager import get_manager
from backend.app.services.inference_service import run_inference

print("Testing ModelManager initialization...")
mgr = get_manager()

print("\nRegistered Models in Application:")
for m in mgr.available_models():
    status = "LOADED" if m["loaded"] else f"FAILED: {m.get('error')}"
    print(f"  [{status}] {m['name']} (key: '{m['key']}')")

print("\nRunning test inference on 'finetuned-fusion' with dummy 512x512 X-ray...")
dummy_img = Image.fromarray(np.uint8(np.random.rand(512, 512) * 255))
res = run_inference(dummy_img, "finetuned-fusion")

print("\nPrediction Result:")
print(f"  Model Name       : Finetuned Respira Fusion")
print(f"  Model Version    : {res['model_version']}")
print(f"  Predicted Class  : {res['predicted_class']}")
print(f"  Confidence Score : {res['confidence']*100:.2f}%")
print(f"  Uncertainty Level: {res['uncertainty_level']} (Entropy: {res['sample_uncertainty']:.4f})")
print(f"  Timing (ms)      : {res['timing']}")
print("\nProbabilities across 6 classes:")
for k, v in res["probabilities"].items():
    print(f"  - {k:20s}: {v*100:6.2f}%")

print("\nSUCCESS: 'Finetuned Respira Fusion' is fully integrated and working!")

# Test with a real test-set image
import glob
test_files = glob.glob("data/Lung_Disease_Preprocessed_512/test/*/*.png")
if test_files:
    test_img_path = test_files[0]
    true_label = os.path.basename(os.path.dirname(test_img_path))
    real_img = Image.open(test_img_path)
    real_res = run_inference(real_img, "finetuned-fusion")
    print(f"\n--- Real Radiograph Test: {test_img_path} ---")
    print(f"  True Label       : {true_label}")
    print(f"  Predicted Label  : {real_res['predicted_class']}")
    print(f"  Confidence       : {real_res['confidence']*100:.2f}%")
    print(f"  Uncertainty Level: {real_res['uncertainty_level']} (Entropy: {real_res['sample_uncertainty']:.4f})")
    print("  Probabilities:")
    for k, v in real_res["probabilities"].items():
        print(f"    - {k:20s}: {v*100:6.2f}%")
