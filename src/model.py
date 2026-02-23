import torch
import torch.nn as nn
from torchvision import models


def get_model():
    # Load pretrained ResNet18
    model = models.resnet18(weights="IMAGENET1K_V1")

    # Modify final layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # 2 classes

    return model