import torch.nn as nn
from torchvision import models


def get_model(backbone="resnet50", unfreeze_from="layer3"):
    """
    Build a binary forensic classifier.

    Args:
        backbone: 'resnet18', 'resnet50', or 'efficientnet_b0'
        unfreeze_from: unfreeze backbone from this layer onward.
            Options for ResNet: 'layer1', 'layer2', 'layer3', 'layer4', 'fc'
            Use 'fc' to freeze the entire backbone (original behaviour).
    """

    if backbone == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V2")
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    else:
        model = models.resnet18(weights="IMAGENET1K_V1")

    # ---------- Freeze / unfreeze strategy ----------
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Then selectively unfreeze from `unfreeze_from` onward
    if backbone.startswith("resnet"):
        layer_order = ["layer1", "layer2", "layer3", "layer4", "fc"]
        if unfreeze_from in layer_order:
            start = layer_order.index(unfreeze_from)
            for name in layer_order[start:]:
                for param in getattr(model, name).parameters():
                    param.requires_grad = True

        # Replace final FC with a stronger classification head
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )
    elif backbone == "efficientnet_b0":
        # Unfreeze last few blocks
        for param in model.features[-3:].parameters():
            param.requires_grad = True
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )

    return model