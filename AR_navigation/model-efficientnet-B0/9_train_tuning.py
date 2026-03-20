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

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


# =========================================================
# 0. 설정값
# =========================================================
TRAIN_DIR = "/home/psy1218/projects/1_pro/images/train"
VAL_DIR   = "/home/psy1218/projects/1_pro/images/val"
TEST_DIR  = "/home/psy1218/projects/1_pro/images/test"

SAVE_PATH = "/home/psy1218/projects/1_pro/efficientnet_b0_best_tuned.pth"

IMAGE_SIZE = (384, 256)   # (H, W)
BATCH_SIZE = 8
NUM_WORKERS = 4

NUM_EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 42

FREEZE_FEATURES = False   # 전체 fine-tuning
EARLY_STOPPING_PATIENCE = 5


# =========================================================
# 1. 시드 고정
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# 2. 로그 함수
# =========================================================
def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def get_gpu_memory_text():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
        return f"GPU mem: alloc={allocated:.2f}GB, reserv={reserved:.2f}GB"
    return "GPU mem: CPU mode"


def get_current_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


# =========================================================
# 3. EXIF 방향 보정 loader
# =========================================================
def exif_loader(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    return img


# =========================================================
# 4. transform
# =========================================================
train_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.RandomRotation(3),   # 5 -> 3으로 약하게 조정
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


# =========================================================
# 5. 데이터셋 / 데이터로더
# =========================================================
def build_dataloaders():
    train_dataset = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=train_transform,
        loader=exif_loader
    )

    val_dataset = datasets.ImageFolder(
        root=VAL_DIR,
        transform=val_test_transform,
        loader=exif_loader
    )

    test_dataset = datasets.ImageFolder(
        root=TEST_DIR,
        transform=val_test_transform,
        loader=exif_loader
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


# =========================================================
# 6. 모델 준비
# =========================================================
def build_model(num_classes, device):
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    if FREEZE_FEATURES:
        for param in model.features.parameters():
            param.requires_grad = False

    model = model.to(device)
    return model


# =========================================================
# 7. epoch 학습
# =========================================================
def train_one_epoch(model, loader, criterion, optimizer, device, epoch_idx, total_epochs):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()

    progress_bar = tqdm(
        loader,
        desc=f"[Train] Epoch {epoch_idx+1}/{total_epochs}",
        leave=False
    )

    for images, labels in progress_bar:
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

        current_loss = running_loss / total
        current_acc = correct / total

        progress_bar.set_postfix({
            "loss": f"{current_loss:.4f}",
            "acc": f"{current_acc:.4f}",
            "lr": f"{get_current_lr(optimizer):.6f}",
            "mem": get_gpu_memory_text()
        })

    epoch_time = time.time() - start_time
    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc, epoch_time


# =========================================================
# 8. epoch 검증/평가
# =========================================================
def evaluate(model, loader, criterion, device, mode="Val"):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []
    all_paths = []

    start_time = time.time()

    progress_bar = tqdm(
        loader,
        desc=f"[{mode}]",
        leave=False
    )

    dataset_samples = loader.dataset.samples
    current_index = 0

    with torch.no_grad():
        for images, labels in progress_bar:
            batch_size = images.size(0)

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            preds = torch.argmax(outputs, dim=1)

            running_loss += loss.item() * batch_size
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

            batch_paths = [dataset_samples[i][0] for i in range(current_index, current_index + batch_size)]
            all_paths.extend(batch_paths)
            current_index += batch_size

            current_loss = running_loss / total
            current_acc = correct / total

            progress_bar.set_postfix({
                "loss": f"{current_loss:.4f}",
                "acc": f"{current_acc:.4f}",
                "mem": get_gpu_memory_text()
            })

    elapsed_time = time.time() - start_time
    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc, all_labels, all_preds, all_paths, elapsed_time


# =========================================================
# 9. confusion matrix 시각화
# =========================================================
def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    plt.figure(figsize=(14, 12))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()


# =========================================================
# 10. 상세 지표용 함수
# =========================================================
def split_zone_direction_from_class_indices(indices, class_names):
    zones = []
    dirs = []

    for idx in indices:
        class_name = class_names[idx]   # 예: '3_W'
        zone, direction = class_name.split('_')
        zones.append(zone)
        dirs.append(direction)

    return zones, dirs


def compute_detailed_metrics(y_true, y_pred, class_names):
    final_acc = accuracy_score(y_true, y_pred)

    true_zones, true_dirs = split_zone_direction_from_class_indices(y_true, class_names)
    pred_zones, pred_dirs = split_zone_direction_from_class_indices(y_pred, class_names)

    zone_acc = accuracy_score(true_zones, pred_zones)
    dir_acc = accuracy_score(true_dirs, pred_dirs)

    matched_zone_mask = [tz == pz for tz, pz in zip(true_zones, pred_zones)]

    true_dirs_when_zone_correct = [td for td, ok in zip(true_dirs, matched_zone_mask) if ok]
    pred_dirs_when_zone_correct = [pd for pd, ok in zip(pred_dirs, matched_zone_mask) if ok]

    if len(true_dirs_when_zone_correct) > 0:
        dir_acc_given_zone_correct = accuracy_score(
            true_dirs_when_zone_correct,
            pred_dirs_when_zone_correct
        )
    else:
        dir_acc_given_zone_correct = 0.0

    return {
        "final_acc": final_acc,
        "zone_acc": zone_acc,
        "dir_acc": dir_acc,
        "dir_acc_given_zone_correct": dir_acc_given_zone_correct
    }


# =========================================================
# 11. 예측 CSV 저장
# =========================================================
def build_prediction_dataframe(y_true, y_pred, file_paths, class_names):
    rows = []

    for true_idx, pred_idx, file_path in zip(y_true, y_pred, file_paths):
        true_class = class_names[true_idx]
        pred_class = class_names[pred_idx]

        true_zone, true_direction = true_class.split('_')
        pred_zone, pred_direction = pred_class.split('_')

        correct_28class = (true_class == pred_class)
        zone_correct = (true_zone == pred_zone)
        direction_correct = (true_direction == pred_direction)

        if correct_28class:
            error_type = "ALL_CORRECT"
        elif (not zone_correct) and direction_correct:
            error_type = "ZONE_ONLY_WRONG"
        elif zone_correct and (not direction_correct):
            error_type = "DIRECTION_ONLY_WRONG"
        else:
            error_type = "BOTH_WRONG"

        rows.append({
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "true_class": true_class,
            "pred_class": pred_class,
            "correct_28class": correct_28class,
            "true_zone": true_zone,
            "pred_zone": pred_zone,
            "zone_correct": zone_correct,
            "true_direction": true_direction,
            "pred_direction": pred_direction,
            "direction_correct": direction_correct,
            "error_type": error_type
        })

    return pd.DataFrame(rows)


def save_prediction_csvs(df, save_dir="/home/psy1218/projects/1_pro"):
    os.makedirs(save_dir, exist_ok=True)

    all_csv_path = os.path.join(save_dir, "test_predictions_all_tuned.csv")
    wrong_csv_path = os.path.join(save_dir, "test_predictions_wrong_only_tuned.csv")

    df.to_csv(all_csv_path, index=False, encoding="utf-8-sig")
    wrong_df = df[df["correct_28class"] == False].copy()
    wrong_df.to_csv(wrong_csv_path, index=False, encoding="utf-8-sig")

    print_section("Prediction CSV Saved")
    print(f"All predictions CSV   : {all_csv_path}")
    print(f"Wrong predictions CSV : {wrong_csv_path}")
    print(f"Total rows            : {len(df)}")
    print(f"Wrong rows            : {len(wrong_df)}")


# =========================================================
# 12. 메인
# =========================================================
def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_section("Environment")
    print(f"Using device         : {device}")
    print(f"Freeze features      : {FREEZE_FEATURES}")
    print(f"Image size           : {IMAGE_SIZE}")
    print(f"Batch size           : {BATCH_SIZE}")
    print(f"Epochs               : {NUM_EPOCHS}")
    print(f"Learning rate        : {LR}")
    print(f"Weight decay         : {WEIGHT_DECAY}")
    print(f"Early stop patience  : {EARLY_STOPPING_PATIENCE}")
    if torch.cuda.is_available():
        print(f"GPU name             : {torch.cuda.get_device_name(0)}")
        print(get_gpu_memory_text())

    print_section("Build Datasets / Loaders")
    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = build_dataloaders()

    class_names = train_dataset.classes
    num_classes = len(class_names)

    print(f"Num classes          : {num_classes}")
    print(f"Train images         : {len(train_dataset)}")
    print(f"Val images           : {len(val_dataset)}")
    print(f"Test images          : {len(test_dataset)}")

    print_section("Build Model")
    model = build_model(num_classes, device)
    criterion = nn.CrossEntropyLoss()

    if FREEZE_FEATURES:
        optimizer = optim.AdamW(model.classifier.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    print("Model ready")
    print(get_gpu_memory_text())

    print_section("Start Training")
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())

    early_stop_counter = 0
    total_train_start = time.time()

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc, train_time = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, NUM_EPOCHS
        )

        val_loss, val_acc, _, _, _, val_time = evaluate(
            model, val_loader, criterion, device, mode=f"Val Epoch {epoch+1}/{NUM_EPOCHS}"
        )

        scheduler.step(val_loss)

        print(
            f"[Epoch {epoch+1:02d}/{NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Train Time: {train_time:.2f}s | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Time: {val_time:.2f}s | "
            f"LR: {get_current_lr(optimizer):.6f} | "
            f"{get_gpu_memory_text()}"
        )

        # best model 저장: val_acc 우선, 동일하면 val_loss 더 좋은 것
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
            print(f"  -> Best model saved! (Val Acc: {best_val_acc:.4f}, Val Loss: {best_val_loss:.4f})")
        else:
            early_stop_counter += 1
            print(f"  -> No improvement. Early stop counter: {early_stop_counter}/{EARLY_STOPPING_PATIENCE}")

        if early_stop_counter >= EARLY_STOPPING_PATIENCE:
            print("  -> Early stopping triggered.")
            break

    total_train_time = time.time() - total_train_start

    print_section("Training Finished")
    print(f"Best Val Accuracy    : {best_val_acc:.4f}")
    print(f"Best Val Loss        : {best_val_loss:.4f}")
    print(f"Saved model path     : {SAVE_PATH}")
    print(f"Total training time  : {total_train_time:.2f}s")

    # best model 로드
    model.load_state_dict(best_model_wts)

    print_section("Test Evaluation")
    test_loss, test_acc, y_true, y_pred, file_paths, test_time = evaluate(
        model, test_loader, criterion, device, mode="Test"
    )

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    detailed_metrics = compute_detailed_metrics(y_true, y_pred, class_names)

    print(f"Test Loss              : {test_loss:.4f}")
    print(f"Test Accuracy          : {test_acc:.4f}")
    print(f"Test Time              : {test_time:.2f}s")
    print(f"Precision (macro)      : {precision_macro:.4f}")
    print(f"Recall (macro)         : {recall_macro:.4f}")
    print(f"F1-score (macro)       : {f1_macro:.4f}")
    print(f"Precision (weighted)   : {precision_weighted:.4f}")
    print(f"Recall (weighted)      : {recall_weighted:.4f}")
    print(f"F1-score (weighted)    : {f1_weighted:.4f}")

    print("\n[Detailed Hierarchical Metrics]")
    print(f"28-class Final Acc     : {detailed_metrics['final_acc']:.4f}")
    print(f"Zone Accuracy          : {detailed_metrics['zone_acc']:.4f}")
    print(f"Direction Accuracy     : {detailed_metrics['dir_acc']:.4f}")
    print(f"Dir Acc | Zone Correct : {detailed_metrics['dir_acc_given_zone_correct']:.4f}")

    prediction_df = build_prediction_dataframe(y_true, y_pred, file_paths, class_names)
    save_prediction_csvs(prediction_df, save_dir="/home/psy1218/projects/1_pro")

    print_section("Classification Report")
    print(classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    ))

    print_section("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    plot_confusion_matrix(cm, class_names, title="EfficientNet-B0 Tuned Test Confusion Matrix")


if __name__ == "__main__":
    main()