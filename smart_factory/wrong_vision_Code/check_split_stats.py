from pathlib import Path
from collections import Counter

# =========================
# 설정
# =========================
ROOT = Path("/home/psy1218/projects/2_pro/YOLOv11")   # 네 경로로 맞추기

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_NAMES = {
    0: "deform",
    1: "hole",
    2: "open",
    3: "retain",
    4: "tear",
    # 필요하면 class id 맞게 수정
}

SPLITS = ["train", "val", "test"]


def get_image_files(img_dir: Path):
    return sorted([
        p for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ])


def parse_label_file(label_path: Path):
    """
    YOLO txt 한 파일에서 class id 목록 추출
    빈 파일이면 []
    """
    if not label_path.exists():
        return []

    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    class_ids = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(parts[0])
            class_ids.append(cls_id)
        except ValueError:
            continue
    return class_ids


def analyze_split(split: str):
    img_dir = ROOT / "images" / split
    lbl_dir = ROOT / "labels" / split

    image_files = get_image_files(img_dir)

    class_counter = Counter()
    normal_images = 0
    labeled_images = 0
    total_boxes = 0

    for img_path in image_files:
        label_path = lbl_dir / f"{img_path.stem}.txt"
        class_ids = parse_label_file(label_path)

        if len(class_ids) == 0:
            normal_images += 1
        else:
            labeled_images += 1
            total_boxes += len(class_ids)
            class_counter.update(class_ids)

    print("=" * 60)
    print(f"[{split.upper()}]")
    print(f"이미지 수            : {len(image_files)}")
    print(f"정상 이미지 수       : {normal_images}")
    print(f"라벨 있는 이미지 수  : {labeled_images}")
    print(f"총 bbox 수           : {total_boxes}")
    print("-" * 60)

    if total_boxes == 0:
        print("bbox가 없습니다.")
        return

    print("클래스별 bbox 개수:")
    for cls_id in sorted(CLASS_NAMES.keys()):
        count = class_counter.get(cls_id, 0)
        ratio = (count / total_boxes * 100) if total_boxes > 0 else 0
        print(f"  {cls_id:>2} ({CLASS_NAMES[cls_id]:<10}) : {count:>5}  ({ratio:6.2f}%)")

    # 정의 안 된 class id가 있는 경우도 출력
    unknown_ids = [cid for cid in class_counter.keys() if cid not in CLASS_NAMES]
    if unknown_ids:
        print("-" * 60)
        print("정의되지 않은 class id 발견:")
        for cid in sorted(unknown_ids):
            count = class_counter[cid]
            ratio = (count / total_boxes * 100) if total_boxes > 0 else 0
            print(f"  {cid:>2} (UNKNOWN)    : {count:>5}  ({ratio:6.2f}%)")


def main():
    for split in SPLITS:
        analyze_split(split)


if __name__ == "__main__":
    main()