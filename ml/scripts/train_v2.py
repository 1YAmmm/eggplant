import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
import timm

import torchvision.transforms as T

# Configuration
DATA_DIR = "data/dataset"  # Adjust to your dataset path
BATCH_SIZE = 32
NUM_CLASSES = 6  # Adjust to your exact number of diseases/pests
EPOCHS = 15
LEARNING_RATE = 1e-3
MODEL_NAME = "mobilenetv3_large_100"  # or "efficientnet_lite0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transformations
train_transforms = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(degrees=30),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Datasets
train_dataset = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_transforms)
val_dataset = datasets.ImageFolder(f"{DATA_DIR}/valid", transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize Model with Pretrained Weights
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)





# Training execution
best_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} | Val Accuracy: {acc*100:.2f}%")
    
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "eggplant_model_best.pth")

        # model.load_state_dict(torch.load("eggplant_model_best.pth"))
        # model.eval()
        # dummy_input = torch.randn(1, 3, 224, 224)
        # torch.onnx.export(model, dummy_input, "eggplant_model.onnx", opset_version=12)