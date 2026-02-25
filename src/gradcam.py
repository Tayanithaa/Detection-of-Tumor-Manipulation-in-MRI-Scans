import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------- LOAD MODEL ----------------
model = models.resnet50(weights=None)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 2),
)

model.load_state_dict(torch.load("forensic_model.pth", map_location=device))
model.to(device)
model.eval()

# ---------------- HOOKS FOR GRAD-CAM ----------------
feature_maps = []
gradients = []

def forward_hook(module, input, output):
    feature_maps.append(output)

def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0])

# Register hooks on final conv layer
target_layer = model.layer4
target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

# ---------------- IMAGE TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---------------- LOAD IMAGE ----------------
image_path = "../Dataset/Processed/test/manipulated/Te-aug-me_1.jpg"
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

# ---------------- FORWARD PASS ----------------
output = model(input_tensor)
pred_class = output.argmax(dim=1).item()

# ---------------- BACKWARD PASS ----------------
model.zero_grad()
output[0, pred_class].backward()

# ---------------- COMPUTE GRAD-CAM ----------------
grad = gradients[0]              # shape: (1, C, H, W)
feature = feature_maps[0]        # shape: (1, C, H, W)

weights = torch.mean(grad, dim=(2, 3), keepdim=True)  # global average pooling
cam = torch.sum(weights * feature, dim=1).squeeze()   # weighted sum

cam = torch.relu(cam)
cam = cam.detach().cpu().numpy()

# Normalize
cam -= cam.min()
cam /= (cam.max() + 1e-8)

# Resize to original image size
cam = cv2.resize(cam, image.size)

# Convert to heatmap
heatmap = np.uint8(255 * cam)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# Overlay heatmap on image
original = np.array(image)
overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

# ---------------- DISPLAY ----------------
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(original)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Grad-CAM")
plt.imshow(overlay)
plt.axis("off")

plt.tight_layout()
plt.show()