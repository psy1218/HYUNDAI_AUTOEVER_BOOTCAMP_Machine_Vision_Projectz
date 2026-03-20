import os
from PIL import Image, ImageOps
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# -------------------------------------------------
# 1. 경로 설정
# -------------------------------------------------
train_dir = "/home/psy1218/projects/1_pro/images/train"
val_dir   = "/home/psy1218/projects/1_pro/images/val"
test_dir  = "/home/psy1218/projects/1_pro/images/test"

# -------------------------------------------------
# 2. EXIF 방향 보정용 커스텀 loader
#    핸드폰 사진 90도 돌아가는 문제 방지
# -------------------------------------------------
def exif_loader(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)   # EXIF 방향 보정
    img = img.convert("RGB")
    return img

# -------------------------------------------------
# 3. transform 정의
# -------------------------------------------------
train_transform = transforms.Compose([
    transforms.Resize((384, 256)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_test_transform = transforms.Compose([
    transforms.Resize((384, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------------------------
# 4. ImageFolder로 데이터셋 만들기
# -------------------------------------------------
train_dataset = datasets.ImageFolder(
    root=train_dir,
    transform=train_transform,
    loader=exif_loader
)

val_dataset = datasets.ImageFolder(
    root=val_dir,
    transform=val_test_transform,
    loader=exif_loader
)

test_dataset = datasets.ImageFolder(
    root=test_dir,
    transform=val_test_transform,
    loader=exif_loader
)

# -------------------------------------------------
# 5. DataLoader 만들기
# -------------------------------------------------
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4
)

# -------------------------------------------------
# 6. 클래스 정보 확인
# -------------------------------------------------
print("클래스 목록:")
print(train_dataset.classes)
print()

print("클래스 개수:", len(train_dataset.classes))
print("train 이미지 수:", len(train_dataset))
print("val 이미지 수:", len(val_dataset))
print("test 이미지 수:", len(test_dataset))
print()

print("class_to_idx:")
print(train_dataset.class_to_idx)

# -------------------------------------------------
# 7. 배치 하나 꺼내서 shape 확인
# -------------------------------------------------
images, labels = next(iter(train_loader))

print()
print("배치 이미지 shape:", images.shape)
print("배치 라벨 shape:", labels.shape)
print("배치 라벨 예시:", labels[:8])