from pathlib import Path
import random
import shutil

# =========================
# 설정
# =========================
ROOT = Path("/home/psy1218/projects/2_pro/YOLOv11")   # 네 경로로 맞추기
VAL_RATIO = 0.2
SEED = 42

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

train_img_dir = ROOT / "images" / "train"
train_lbl_dir = ROOT / "labels" / "train"
val_img_dir = ROOT / "images" / "val"
val_lbl_dir = ROOT / "labels" / "val"

val_img_dir.mkdir(parents=True, exist_ok=True)
val_lbl_dir.mkdir(parents=True, exist_ok=True)

# =========================
# train 이미지 목록 수집
# =========================
image_files = sorted([
    p for p in train_img_dir.iterdir()
    if p.is_file() and p.suffix.lower() in IMG_EXTS
])

if len(image_files) == 0:
    raise ValueError(f"train 이미지가 없습니다: {train_img_dir}")

# =========================
# 랜덤 분할
# =========================
random.seed(SEED)
random.shuffle(image_files)

num_val = int(len(image_files) * VAL_RATIO)
val_candidates = image_files[:num_val]

print(f"[INFO] 전체 train 이미지 수: {len(image_files)}")
print(f"[INFO] val로 이동할 이미지 수: {num_val}")

# =========================
# 이동
# =========================
moved_count = 0
missing_label_count = 0

for img_path in val_candidates:
    label_path = train_lbl_dir / f"{img_path.stem}.txt"

    # 이미지 이동
    dst_img_path = val_img_dir / img_path.name
    shutil.move(str(img_path), str(dst_img_path))

    # 라벨 이동 (있으면)
    if label_path.exists():
        dst_lbl_path = val_lbl_dir / label_path.name
        shutil.move(str(label_path), str(dst_lbl_path))
    else:
        # 정상 이미지일 가능성
        missing_label_count += 1

    moved_count += 1

print(f"[DONE] 이동 완료")
print(f"[INFO] 이동된 이미지 수: {moved_count}")
print(f"[INFO] 라벨 없는(정상 가능) 이미지 수: {missing_label_count}")
print(f"[INFO] 남은 train 이미지 수: {len(list(train_img_dir.iterdir()))}")
print(f"[INFO] 현재 val 이미지 수: {len(list(val_img_dir.iterdir()))}")