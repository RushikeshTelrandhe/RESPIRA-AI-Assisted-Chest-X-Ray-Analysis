import os
import sys
import glob
sys.path.append(os.path.abspath("."))
from PIL import Image
from backend.app.services.inference_service import run_inference

tb_images = glob.glob("data/Lung_Disease_Preprocessed_512/test/Tuberculosis/*.png")[:5]
print(f"Testing {len(tb_images)} real TB test set images:")

for p in tb_images:
    img = Image.open(p)
    res = run_inference(img, "finetuned-fusion")
    fn = os.path.basename(p)
    pred = res["predicted_class"]
    conf = res["confidence"] * 100
    tb_p = res["probabilities"]["Tuberculosis"] * 100
    print(f"File: {fn:25s} -> Pred: {pred:20s} (Conf: {conf:5.1f}%, TB Prob: {tb_p:5.1f}%)")
