import os
import random
import shutil

train_root = "/home/psy1218/projects/1_pro/new_images/train"
val_root = "/home/psy1218/projects/1_pro/new_images/val"
val_ratio = 0.2
seed = 42

random.seed(seed)
valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

os.makedirs(val_root, exist_ok=True)

for class_name in sorted(os.listdir(train_root)):
    train_class_path = os.path.join(train_root, class_name)

    if not os.path.isdir(train_class_path):
        continue

    val_class_path = os.path.join(val_root, class_name)
    os.makedirs(val_class_path, exist_ok=True)

    image_files = [
        f for f in os.listdir(train_class_path)
        if os.path.isfile(os.path.join(train_class_path, f))
        and f.lower().endswith(valid_ext)
    ]

    if len(image_files) == 0:
        print(f"[{class_name}] 이미지 없음 - 건너뜀")
        continue

    image_files.sort()
    random.shuffle(image_files)

    val_count = int(len(image_files) * val_ratio)
    val_files = image_files[:val_count]

    print(f"\n[{class_name}] 총 {len(image_files)}장 -> val로 {val_count}장 이동")

    for fname in val_files:
        src = os.path.join(train_class_path, fname)
        dst = os.path.join(val_class_path, fname)
        shutil.move(src, dst)

print("\ntrain -> val 이동 완료")