import os
import sys
sys.path.append(os.path.abspath("."))

from PIL import Image
import numpy as np

from backend.app.services.model_manager import get_manager
from backend.app.services.inference_service import run_inference
from backend.app.services.explainability_service import gradcam_for, vit_attention_for
from app.backend import config as ml_config

def main():
    print("Testing End-to-End Finetuned Respira Fusion Explainability...")
    mgr = get_manager()
    print("Manager ready:", mgr.ready)
    print("Finetuned model ready:", mgr.finetuned_fusion is not None)

    # Test with tuber.jpeg
    img_path = "backend_storage/analyses/6e0fee28330044f1934cde13a10e9083/original.png"
    if not os.path.exists(img_path):
        print("Image not found:", img_path)
        return

    img = Image.open(img_path)
    print(f"Loaded radiograph: {img.size}")

    # 1. Run inference with finetuned-fusion
    res = run_inference(img, "finetuned-fusion")
    pred_class = res["predicted_class"]
    pred_idx = ml_config.CLASS_NAMES.index(pred_class)
    print(f"Inference predicted: {pred_class} (idx {pred_idx}) with confidence {res['confidence']*100:.2f}%")
    print(f"Uncertainty: {res['uncertainty_level']} ({res['sample_uncertainty']:.4f})")

    # 2. Run Grad-CAM with finetuned-fusion
    grad_res = gradcam_for(img, pred_idx, "finetuned-fusion")
    print(f"Grad-CAM generated in {grad_res['explainability_ms']:.1f}ms")
    print("Target class:", grad_res["target_class"])
    print("Heatmap URL prefix:", grad_res["heatmap"][:30])
    print("Overlay URL prefix:", grad_res["overlay"][:30])

    # Decode and check dimensions
    import base64, io
    overlay_bytes = base64.b64decode(grad_res["overlay"].split(",")[1])
    overlay_img = Image.open(io.BytesIO(overlay_bytes))
    print(f"Grad-CAM overlay dimensions: {overlay_img.size}")
    assert overlay_img.size == (512, 512), f"Expected (512, 512), got {overlay_img.size}"

    overlay_arr = np.array(overlay_img)
    # Check shoulder region (top right armpit):
    # in 512x512: y in [50, 130], x in [360, 490]
    shoulder_red = overlay_arr[50:130, 360:490, 0]
    shoulder_green = overlay_arr[50:130, 360:490, 1]
    shoulder_blue = overlay_arr[50:130, 360:490, 2]
    # In JET colormap, red has high R, low G, low B.
    # In unmasked original radiograph, shoulder soft tissue is bone/grayscale (R ~= G ~= B).
    # If there is a false red hotspot, R >> B.
    red_bias = (shoulder_red.astype(float) - shoulder_blue.astype(float)).mean()
    print(f"Shoulder Red-Blue bias: {red_bias:.2f} (Should be near 0; old buggy overlay was > 120)")
    assert red_bias < 20, f"Red artifact still present on shoulder! Bias: {red_bias}"

    # 3. Run ViT Attention with finetuned-fusion
    vit_res = vit_attention_for(img, pred_idx, "finetuned-fusion")
    print(f"ViT Attention generated in {vit_res['explainability_ms']:.1f}ms")
    vit_overlay_bytes = base64.b64decode(vit_res["overlay"].split(",")[1])
    vit_overlay_img = Image.open(io.BytesIO(vit_overlay_bytes))
    print(f"ViT overlay dimensions: {vit_overlay_img.size}")
    assert vit_overlay_img.size == (512, 512), f"Expected (512, 512), got {vit_overlay_img.size}"

    # 4. Save sample results for inspection
    os.makedirs("outputs/verified_explainability", exist_ok=True)
    overlay_img.save("outputs/verified_explainability/finetuned_gradcam_512.png")
    vit_overlay_img.save("outputs/verified_explainability/finetuned_vit_512.png")
    print("Saved verified 512x512 heatmaps to outputs/verified_explainability/")
    print("\nALL VERIFICATION TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
