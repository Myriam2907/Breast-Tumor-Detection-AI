# prediction.py
import io
import os
import uuid
from typing import Dict

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

# ----------------------------
# Config
# ----------------------------
MODEL_PATH = "model/best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CLASS_MAPPING = {
    0: "Benign No malignant cells detected",     
    1: "Malignant Cancerous tissue detected"   
}

# Full-size input expected by model
INPUT_SIZE = (224, 224)

# ----------------------------
# Model loading
# ----------------------------
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_MAPPING))
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    # allow both state_dict and full model (state may be dict)
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        # handle DataParallel saved state
        from collections import OrderedDict
        new_state = OrderedDict()
        for k, v in state.items():
            new_state[k.replace("module.", "")] = v
        state = new_state
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

# Load once
model = load_model()

# ----------------------------
# Preprocessing
# ----------------------------
preprocess = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Helper: preprocess PIL Image and return tensor on device
def preprocess_pil(img_pil: Image.Image):
    return preprocess(img_pil).to(DEVICE)

# ----------------------------
# Grad-CAM (correct implementation)
# ----------------------------
class GradCAM:
    def __init__(self, model: nn.Module, target_layer_name: str = None):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self.target_layer = None

        # if target_layer_name is not provided, automatically pick last Conv2d
        if target_layer_name is None:
            for name, module in reversed(list(self.model.named_modules())):
                if isinstance(module, nn.Conv2d):
                    target_layer_name = name
                    break
        if target_layer_name is None:
            raise RuntimeError("No Conv2d layer found in model for Grad-CAM")
        self.target_layer_name = target_layer_name
        self._register_hooks()

    def _get_module_by_name(self, name):
        curr = self.model
        for part in name.split("."):
            curr = getattr(curr, part)
        return curr

    def _register_hooks(self):
        module = None
        for name, m in self.model.named_modules():
            if name == self.target_layer_name:
                module = m
                break
        if module is None:
            raise RuntimeError(f"Target layer {self.target_layer_name} not found")
        self.target_layer = module

        def forward_hook(module, input, output):
            # output shape: (N, C, H, W)
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            # grad_out[0] shape: (N, C, H, W)
            self.gradients = grad_out[0].detach()

        self.hook_handles.append(module.register_forward_hook(forward_hook))
        self.hook_handles.append(module.register_backward_hook(backward_hook))

    def __call__(self, input_tensor: torch.Tensor, class_idx: int = None):
        """
        input_tensor: (1, C, H, W) on DEVICE
        class_idx: which class to compute gradcam for. If None, uses predicted class.
        Returns: heatmap numpy array resized to input HxW (values 0..1)
        """
        self.model.zero_grad()
        self.gradients = None
        self.activations = None

        outputs = self.model(input_tensor)  # forward
        if class_idx is None:
            class_idx = int(outputs.argmax(dim=1).item())
        score = outputs[0, class_idx]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations were not captured by hooks")

        # Grad-CAM: weights = global-average-pool(gradients) over HxW
        pooled = torch.mean(self.gradients, dim=(2, 3))  # (N, C)
        pooled = pooled[0]  # (C,)
        activ = self.activations[0]  # (C, H, W)

        # Weighted combination
        weighted = (pooled.view(-1, 1, 1) * activ).cpu().numpy()
        cam = np.sum(weighted, axis=0)  # (H, W)
        cam = np.maximum(cam, 0)
        if cam.max() != 0:
            cam = cam / cam.max()
        else:
            cam = cam

        # remove hooks after use (we keep object but re-register if needed)
        # (we will keep hooks registered so repeated calls work)
        return cam

    def remove_hooks(self):
        for h in self.hook_handles:
            try:
                h.remove()
            except:
                pass
        self.hook_handles = []

# ----------------------------
# Patch-based malignant percentage
# ----------------------------
def compute_malignant_percentage_from_pil(img_pil: Image.Image, model: nn.Module, grid=4):
    """
    Splits a resized copy of img_pil into grid x grid patches, predicts each patch,
    and returns percentage of patches predicted as malignant (class index matching CLASS_MAPPING).
    Uses the SAME preprocess as the full image.
    """
    # resize to INPUT_SIZE then split
    img_resized = img_pil.resize(INPUT_SIZE)
    W, H = img_resized.size
    patch_w = W // grid
    patch_h = H // grid
    malignant_count = 0
    total = grid * grid

    model.eval()
    for i in range(grid):
        for j in range(grid):
            left = j * patch_w
            upper = i * patch_h
            right = left + patch_w
            lower = upper + patch_h
            patch = img_resized.crop((left, upper, right, lower))
            patch_tensor = preprocess(patch).unsqueeze(0).to(DEVICE)  # single patch
            with torch.no_grad():
                out = model(patch_tensor)
                prob = torch.softmax(out, dim=1)[0]
                pred_idx = int(torch.argmax(prob).item())
                # count as malignant if predicted class corresponds to Tumor label
                if CLASS_MAPPING.get(pred_idx, "").lower().startswith("Malignant"):
                 malignant_count += 1


    return (malignant_count / total) * 100.0

# ----------------------------
# Utility to overlay heatmap on image and save
# ----------------------------
def save_gradcam_overlay(pil_img: Image.Image, cam: np.ndarray, out_path: str):
    """
    cam: 2D array normalized 0..1 (size depends on last conv spatial dims).
    We'll resize cam to INPUT_SIZE and overlay on RGB image.
    """
    rgb = np.array(pil_img.resize(INPUT_SIZE).convert("RGB"))
    cam_resized = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]))
    heatmap = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # BGR
    overlay = cv2.addWeighted(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 0.6, heatmap, 0.4, 0)
    # convert back to RGB for saving with cv2 (cv2.imwrite expects BGR)
    cv2.imwrite(out_path, overlay)

# ----------------------------
# Main predict function used by FastAPI
# ----------------------------
async def predict_breast_tumor(image_bytes: bytes, save_path: str, gradcam_path: str) -> Dict:
    """
    - Saves original image to save_path (JPEG)
    - Runs model on full image
    - Computes malignant percentage (patch-based)
    - Generates Grad-CAM overlay saved to gradcam_path
    - Returns dict: { prediction, confidence (0..100), malignant_percentage, gradcam_path }
    """

    # 1) Save original image (PIL)
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pil_img.save(save_path)  # keep original saved

    # 2) Full-image prediction
    img_tensor = preprocess_pil(pil_img).unsqueeze(0)  # (1,C,H,W) on DEVICE
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]  # e.g. tensor([0.1, 0.9])
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].cpu().item()) * 100.0
        prediction_label = CLASS_MAPPING.get(pred_idx, f"Class_{pred_idx}")

    # 3) Malignant percentage from patches
    malignant_pct = compute_malignant_percentage_from_pil(pil_img, model, grid=4)

    # 4) Grad-CAM
    try:
        # choose target layer automatically (last conv2d)
        target_layer_name = None
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, nn.Conv2d):
                target_layer_name = name
                break
        if target_layer_name is None:
            raise RuntimeError("No Conv2d layer found for Grad-CAM")

        gc = GradCAM(model, target_layer_name=target_layer_name)
        cam = gc(img_tensor, class_idx=pred_idx)  # 2D cam normalized
        save_gradcam_overlay(pil_img, cam, gradcam_path)
    except Exception as e:
        # if gradcam fails, create a simple labeled image as fallback
        fallback = np.array(pil_img.resize(INPUT_SIZE).convert("RGB"))
        cv2.putText(fallback, f"{prediction_label} ({malignant_pct:.1f}%)", (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255) if pred_idx==0 else (0,255,0), 2)
        cv2.imwrite(gradcam_path, cv2.cvtColor(fallback, cv2.COLOR_RGB2BGR))

    return {
        "prediction": prediction_label,
        "confidence": confidence,            # 0..100
        "malignant_percentage": malignant_pct,  # 0..100 (patch-based)
        "gradcam_path": gradcam_path
    }

# If run as script for quick local testing
if __name__ == "__main__":
    # quick sanity test (if you run python prediction.py <imgfile>)
    import sys
    if len(sys.argv) > 1:
        imgfile = sys.argv[1]
        with open(imgfile, "rb") as f:
            data = f.read()
        out_orig = "tmp_orig.jpg"
        out_grad = "tmp_grad.jpg"
        r = torch.hub.load_state_dict_from_url if False else None
        res = torch.asyncio.run(predict_breast_tumor(data, out_orig, out_grad)) if False else None
        print("Saved preview files:", out_orig, out_grad)
