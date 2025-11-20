# utils/model_utils.py
import torch
import torch.nn as nn
from torchvision import models
import cv2
import numpy as np
from config import MODEL_PATH, IMG_SIZE, CONF_THRESHOLD, CLASS_LABELS

def load_model(device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading model on:", device)

    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_LABELS))

    state_dict = torch.load(MODEL_PATH, map_location=device)
    # If state_dict saved with model.state_dict() directly, this works.
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device

def preprocess(img):
    # img: BGR numpy array (cv2)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    img_resized = img_resized.astype(np.float32) / 255.0
    img_chw = np.transpose(img_resized, (2, 0, 1))
    tensor = torch.tensor(img_chw, dtype=torch.float32).unsqueeze(0)
    return tensor

def predict_breast_tumor(img, model, device, CONF_THRESHOLD=CONF_THRESHOLD, CLASS_LABELS=CLASS_LABELS):
    img_tensor = preprocess(img).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])

    if confidence < CONF_THRESHOLD:
        return "Needs Review", confidence
    return CLASS_LABELS[pred_idx], confidence

# Grad-CAM (simple implementation)
def get_gradcam(img, model, target_layer_name=None, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    tensor = preprocess(img).to(device)

    # auto-detect last conv
    if target_layer_name is None:
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, nn.Conv2d):
                target_layer_name = name
                break

    features = []
    gradients = []

    def forward_hook(module, input, output):
        features.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # register hooks
    for name, module in model.named_modules():
        if name == target_layer_name:
            module.register_forward_hook(forward_hook)
            module.register_backward_hook(backward_hook)

    outputs = model(tensor)
    class_idx = int(outputs.argmax().item())

    model.zero_grad()
    outputs[0, class_idx].backward()

    grads = gradients[0].cpu().numpy()[0]
    fmap = features[0].cpu().detach().numpy()[0]

    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * fmap[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return superimposed
