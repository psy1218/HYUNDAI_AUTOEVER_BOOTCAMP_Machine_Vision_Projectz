import os
import time
import copy
import random
import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)


# =============================
# 설정
# =============================
TRAIN_DIR = "/home/psy1218/projects/1_pro/images/train"
VAL_DIR   = "/home/psy1218/projects/1_pro/images/val"
TEST_DIR  = "/home/psy1218/projects/1_pro/images/test"

SAVE_PATH = "/home/psy1218/projects/1_pro/exp1_baseline_simple_best.pth"

IMAGE_SIZE = (384, 256)
BATCH_SIZE = 8
NUM_WORKERS = 4

NUM_EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5
FREEZE_FEATURES = False
SEED = 42


# =============================
# 시드 고정
# =============================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================
# EXIF 보정
# =============================
def exif_loader(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    return img


# =============================
# transform
# =============================
train_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.RandomRotation(3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_test_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =============================
# 데이터로더
# =============================
def build_dataloaders():
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform, loader=exif_loader)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_test_transform, loader=exif_loader)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=val_test_transform, loader=exif_loader)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


# =============================
# 모델
# =============================
def build_model(num_classes, device):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    if FREEZE_FEATURES:
        for param in model.features.parameters():
            param.requires_grad = False

    return model.to(device)


# =============================
# 학습 1 epoch
# =============================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=1)

        running_loss += loss.item() * images.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


# =============================
# 평가
# =============================
def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            preds = torch.argmax(outputs, dim=1)

            running_loss += loss.item() * images.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    return running_loss / total, correct / total, y_true, y_pred


# =============================
# 구역/방향 분리 지표
# =============================
def split_zone_direction(indices, class_names):
    zones = []
    dirs = []

    for idx in indices:
        class_name = class_names[idx]   # 예: 3_W
        zone, direction = class_name.split("_")
        zones.append(zone)
        dirs.append(direction)

    return zones, dirs


def compute_detailed_metrics(y_true, y_pred, class_names):
    true_zones, true_dirs = split_zone_direction(y_true, class_names)
    pred_zones, pred_dirs = split_zone_direction(y_pred, class_names)

    zone_acc = accuracy_score(true_zones, pred_zones)
    dir_acc = accuracy_score(true_dirs, pred_dirs)

    zone_correct_mask = [tz == pz for tz, pz in zip(true_zones, pred_zones)]

    true_dirs_when_zone_correct = [td for td, ok in zip(true_dirs, zone_correct_mask) if ok]
    pred_dirs_when_zone_correct = [pd for pd, ok in zip(pred_dirs, zone_correct_mask) if ok]

    if len(true_dirs_when_zone_correct) > 0:
        dir_acc_given_zone_correct = accuracy_score(true_dirs_when_zone_correct, pred_dirs_when_zone_correct)
    else:
        dir_acc_given_zone_correct = 0.0

    return zone_acc, dir_acc, dir_acc_given_zone_correct


# =============================
# 메인
# =============================
def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = build_dataloaders()
    class_names = train_dataset.classes
    num_classes = len(class_names)

    print("Num classes :", num_classes)
    print("Train images:", len(train_dataset))
    print("Val images  :", len(val_dataset))
    print("Test images :", len(test_dataset))

    model = build_model(num_classes, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    early_stop_counter = 0

    for epoch in range(NUM_EPOCHS):
        start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(
            f"[Epoch {epoch+1:02d}/{NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
            f"Time: {time.time()-start:.2f}s"
        )

        improved = False
        if val_acc > best_val_acc:
            improved = True
        elif val_acc == best_val_acc and val_loss < best_val_loss:
            improved = True

        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, SAVE_PATH)
            early_stop_counter = 0
            print("-> Best model saved")
        else:
            early_stop_counter += 1
            print(f"-> No improvement ({early_stop_counter}/{EARLY_STOPPING_PATIENCE})")

        if early_stop_counter >= EARLY_STOPPING_PATIENCE:
            print("-> Early stopping")
            break

    # best model 로드
    model.load_state_dict(best_model_wts)

    # test 평가
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    zone_acc, dir_acc, dir_acc_given_zone_correct = compute_detailed_metrics(
        y_true, y_pred, class_names
    )

    cm = confusion_matrix(y_true, y_pred)

    print("\n===== TEST RESULT =====")
    print(f"Test Loss              : {test_loss:.4f}")
    print(f"Test Accuracy          : {test_acc:.4f}")
    print(f"Precision (macro)      : {precision_macro:.4f}")
    print(f"Recall (macro)         : {recall_macro:.4f}")
    print(f"F1-score (macro)       : {f1_macro:.4f}")
    print(f"Precision (weighted)   : {precision_weighted:.4f}")
    print(f"Recall (weighted)      : {recall_weighted:.4f}")
    print(f"F1-score (weighted)    : {f1_weighted:.4f}")
    print(f"Zone Accuracy          : {zone_acc:.4f}")
    print(f"Direction Accuracy     : {dir_acc:.4f}")
    print(f"Dir Acc | Zone Correct : {dir_acc_given_zone_correct:.4f}")

    print("\n===== CLASSIFICATION REPORT =====")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    print("\n===== CONFUSION MATRIX =====")
    print(cm)


if __name__ == "__main__":
    main()