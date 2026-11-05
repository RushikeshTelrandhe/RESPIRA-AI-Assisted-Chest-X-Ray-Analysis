import os
import sys
sys.path.append(os.path.abspath("."))

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from backend.app.services.model_manager import get_manager
from app.backend.preprocessing import preprocess_pil
from app.backend import config as ml_config

def main():
    mgr = get_manager()
    print("Device:", mgr.device)
    model = mgr.finetuned_fusion
    if model is None:
        print("Error: finetuned_fusion is None!")
        return

    img_path = "backend_storage/analyses/6e0fee28330044f1934cde13a10e9083/original.png"
    if not os.path.exists(img_path):
        print("Image not found:", img_path)
        return

    img = Image.open(img_path)
    final_arr, _ = preprocess_pil(img, image_size=512)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(Image.fromarray(final_arr)).unsqueeze(0).to(mgr.device)

    # 1. Test forward pass
    out = model(tensor, return_intermediates=True)
    logits = out["logits"]
    probs = torch.softmax(logits, dim=-1)[0]
    print("\nModel Predictions on test image:")
    for name, p in zip(ml_config.CLASS_NAMES, probs):
        print(f"  {name:20s}: {p.item()*100:.2f}%")

    pred_idx = int(torch.argmax(probs).item())
    print(f"Top predicted index: {pred_idx} ({ml_config.CLASS_NAMES[pred_idx]})")

    # 2. Test Grad-CAM on model.cnn_backbone[-1]
    activations = []
    gradients = []

    def fwd_hook(module, inp, output):
        activations.append(output)

    def bwd_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    target_layer = model.cnn_backbone[-1]
    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    model.zero_grad()
    logits = model(tensor)["logits"]
    target_score = logits[0, pred_idx]
    target_score.backward()

    h1.remove()
    h2.remove()

    act = activations[0] # [1, 1280, 16, 16]
    grad = gradients[0]  # [1, 1280, 16, 16]
    print(f"\nGrad-CAM captured: Act shape {act.shape}, Grad shape {grad.shape}")

    weights = grad.mean(dim=(2, 3), keepdim=True) # [1, 1280, 1, 1]
    cam = torch.relu((weights * act).sum(dim=1, keepdim=True)) # [1, 1, 16, 16]
    cam = torch.nn.functional.interpolate(cam, size=(512, 512), mode="bilinear", align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()

    print(f"CAM shape: {cam.shape}, min: {cam.min():.4f}, max: {cam.max():.4f}")

    # Check activation on shoulder (e.g. y=80, x=208 or normalized top right area)
    # y in [0..120], x in [160..260]
    shoulder_val = cam[60:120, 160:260].mean()
    lung_val = cam[180:400, 100:400].mean()
    print(f"Mean CAM in shoulder/armpit region [60:120, 160:260]: {shoulder_val:.4f}")
    print(f"Mean CAM in bilateral lung field [180:400, 100:400]: {lung_val:.4f}")

    # 3. Test ViT Disease Attention map
    disease_att = out["disease_att"] # [1, 6, 1280]
    print("disease_att shape:", disease_att.shape)
    # The first 256 tokens are CNN tokens (16x16), next 1024 are ViT tokens (32x32)
    vit_att = disease_att[0, pred_idx, 256:].reshape(32, 32).detach().cpu().numpy()
    vit_att = vit_att - vit_att.min()
    if vit_att.max() > 0:
        vit_att = vit_att / vit_att.max()
    print("ViT attention map (32x32) normalized successfully, min:", vit_att.min(), "max:", vit_att.max())

if __name__ == "__main__":
    main()
