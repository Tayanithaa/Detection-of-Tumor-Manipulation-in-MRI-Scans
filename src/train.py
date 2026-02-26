import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import copy

from model import get_model
from data_loader import get_dataloaders

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

# ---------------- DATA ----------------
train_loader, test_loader = get_dataloaders(
    "../Dataset/Processed/train",
    "../Dataset/Processed/test",
    batch_size=8
)

# ---------------- MODEL ----------------
# Use ResNet-50 with partial fine-tuning (unfreeze from layer3 onward)
model = get_model(backbone="resnet50", unfreeze_from="layer3").to(device)

# ---------------- LOSS & OPTIMIZER ----------------
criterion = nn.CrossEntropyLoss()

# Differential learning rates: backbone layers get a smaller LR
backbone_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and "fc" not in n]
head_params = [p for n, p in model.named_parameters()
               if p.requires_grad and "fc" in n]

optimizer = optim.Adam([
    {"params": backbone_params, "lr": 1e-5},   # fine-tune slowly
    {"params": head_params,     "lr": 1e-3},   # train head faster
], weight_decay=1e-4)

# Cosine annealing scheduler
num_epochs = 20
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)

# ---------------- TRAINING WITH VALIDATION TRACKING ----------------
best_val_acc = 0.0
best_model_state = None
patience = 7
no_improve = 0

for epoch in range(num_epochs):
    # ---- Train ----
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    print(f"\nEpoch [{epoch+1}/{num_epochs}]  LR: {scheduler.get_last_lr()}")

    for images, labels in tqdm(train_loader, desc="Train"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total
    print(f"  Train Loss: {running_loss/len(train_loader):.4f}  Acc: {train_acc:.2f}%")

    # ---- Validate ----
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Val"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total
    print(f"  Val   Loss: {val_loss/len(test_loader):.4f}  Acc: {val_acc:.2f}%")

    # ---- Checkpoint best model ----
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = copy.deepcopy(model.state_dict())
        no_improve = 0
        print(f"  >> New best model (val acc: {val_acc:.2f}%)")
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"  Early stopping after {patience} epochs without improvement.")
            break

    scheduler.step()

# ---------------- RESTORE BEST MODEL ----------------
if best_model_state is not None:
    model.load_state_dict(best_model_state)

# ---------------- FINAL EVALUATION ----------------
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

print("\nClassification Report:")
print(classification_report(
    all_labels,
    all_preds,
    target_names=["Real", "Manipulated"]
))

print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")

# ---------------- SAVE MODEL ----------------
torch.save(model.state_dict(), "forensic_model.pth")
print("Model saved as forensic_model.pth")