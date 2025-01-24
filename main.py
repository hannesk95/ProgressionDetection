import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from monai.networks.nets import DenseNet121
from sklearn.metrics import (
    matthews_corrcoef, balanced_accuracy_score, f1_score, roc_auc_score
)
from tqdm import tqdm
from monai.transforms import PadListDataCollate
from dataset import BurdenkoLumiereDataset
from utils import track_training_progress, seed_everything
from model import ConvNeXt3D

#-------------------------------
# Initialize Project
#-------------------------------
torch.cuda.empty_cache()
seed_everything(28)

# -------------------------------
# Dataset Preparation
# -------------------------------
train_dataset = BurdenkoLumiereDataset(split="train")
val_dataset = BurdenkoLumiereDataset(split="val")
test_dataset = BurdenkoLumiereDataset(split="test")

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, collate_fn=PadListDataCollate())
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=PadListDataCollate())
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=PadListDataCollate())

# -------------------------------
# Model Definition
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = DenseNet121(spatial_dims=3, in_channels=8, out_channels=1).to(device)
model = ConvNeXt3D(in_chans=8, num_classes=1).to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# -------------------------------
# Metrics
# -------------------------------
def calculate_metrics(y_true, y_pred):
    y_pred_bin = (y_pred >= 0.5).astype(int)  # Threshold at 0.5
    metrics = {
        "MCC": matthews_corrcoef(y_true, y_pred_bin),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred_bin),
        "F1-Score": f1_score(y_true, y_pred_bin),
        "AUROC": roc_auc_score(y_true, y_pred),
    }
    return metrics

# -------------------------------
# Training and Validation Loops
# -------------------------------
num_epochs = 100
best_metric = -1
best_metric_epoch = -1

train_loss_list = []
val_loss_list = []
train_auc_list = []
val_auc_list = []
train_bacc_list = []
val_bacc_list = []
train_mcc_list = []
val_mcc_list = []
train_f1_list = []
val_f1_list = []
lr_list  = []

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    # Training loop
    model.train()
    train_loss = 0
    all_labels = []
    all_preds = []
    for batch_data in tqdm(train_loader, desc="Training"):
        inputs, labels = batch_data[0].to(device, dtype=torch.float32), batch_data[1].to(device, dtype=torch.float32)
        optimizer.zero_grad()
        outputs = model(inputs).squeeze(dim=1)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())

    train_loss /= len(train_loader)
    print(f"Train Loss: {train_loss:.4f}")

    # Calculate metrics
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    metrics = calculate_metrics(all_labels, all_preds)

    train_loss_list.append(train_loss)
    train_bacc_list.append(metrics["Balanced Accuracy"])
    train_auc_list.append(metrics["AUROC"])
    train_mcc_list.append(metrics["MCC"])
    train_f1_list.append(metrics["F1-Score"])
    lr_list.append(optimizer.param_groups[0]['lr'])

    # Validation loop
    model.eval()
    val_loss = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc="Validation"):
            inputs, labels = batch_data[0].to(device, dtype=torch.float32), batch_data[1].to(device, dtype=torch.float32)
            outputs = model(inputs).squeeze(dim=1)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(torch.sigmoid(outputs).cpu().numpy())

    val_loss /= len(val_loader)
    print(f"Val Loss: {val_loss:.4f}")

    # Calculate metrics
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    metrics = calculate_metrics(all_labels, all_preds)

    val_loss_list.append(val_loss)
    val_bacc_list.append(metrics["Balanced Accuracy"])
    val_auc_list.append(metrics["AUROC"])
    val_mcc_list.append(metrics["MCC"])
    val_f1_list.append(metrics["F1-Score"])

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    # Save the best model based on AUROC
    if metrics["AUROC"] > best_metric:
        best_metric = metrics["AUROC"]
        best_metric_epoch = epoch + 1
        torch.save(model.state_dict(), "./results/best_model.pth")
        print("Saved Best Model!")

    track_training_progress(train_loss_list, val_loss_list,
                            train_auc_list, val_auc_list,
                            train_bacc_list, val_bacc_list,
                            train_mcc_list, val_mcc_list,
                            train_f1_list, val_f1_list,
                            lr_list, result_dir = "./results/")

print(f"Best Metric (AUROC): {best_metric:.4f} at Epoch {best_metric_epoch}")

# Test loop
model.load_state_dict(torch.load("./results/best_model.pth"))
model.eval()
test_loss = 0
all_labels = []
all_preds = []
with torch.no_grad():
    for batch_data in tqdm(test_loader, desc="Test"):
        inputs, labels = batch_data[0].to(device, dtype=torch.float32), batch_data[1].to(device, dtype=torch.float32)
        outputs = model(inputs).squeeze(dim=1)
        loss = loss_fn(outputs, labels)
        val_loss += loss.item()

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(torch.sigmoid(outputs).cpu().numpy())

test_loss /= len(test_loader)
print(f"Test Loss: {test_loss:.4f}")

# Calculate metrics
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
metrics = calculate_metrics(all_labels, all_preds)

for metric_name, metric_value in metrics.items():
    print(f"{metric_name}: {metric_value:.4f}")
