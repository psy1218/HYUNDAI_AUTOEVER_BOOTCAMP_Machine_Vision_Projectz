import os
import glob
from pathlib import Path

import cv2

# =========================
# 사용자 설정
# =========================
IMAGE_DIR = "/home/psy1218/projects/2_pro/YOLOv11/images/train"
LABEL_DIR = "/home/psy1218/projects/2_pro/YOLOv11/labels/train"
OUTPUT_DIR = "/home/psy1218/projects/2_pro/YOLOv11/check_overlay_train"

CLASS_NAMES = {
    0: "deform",
    1: "hole",
    2: "open",
    3: "retain",
    4: "tear",
}

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]
os.makedirs(OUTPUT_DIR, exist_ok=True)


def yolo_to_xyxy(line, img_w, img_h):
    parts = line.strip().split()
    if len(parts) != 5:
        return None

    cls_id, xc, yc, w, h = map(float, parts)
    cls_id = int(cls_id)

    x1 = int((xc - w / 2) * img_w)
    y1 = int((yc - h / 2) * img_h)
    x2 = int((xc + w / 2) * img_w)
    y2 = int((yc + h / 2) * img_h)

    return cls_id, x1, y1, x2, y2


def main():
    image_paths = []
    for ext in IMG_EXTS:
        image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, f"*{ext}")))
    image_paths = sorted(image_paths)

    print(f"이미지 수: {len(image_paths)}")

    for i, image_path in enumerate(image_paths):
        image_name = os.path.basename(image_path)
        stem = Path(image_path).stem
        label_path = os.path.join(LABEL_DIR, stem + ".txt")

        img = cv2.imread(image_path)
        if img is None:
            print(f"[실패] 이미지 읽기 실패: {image_name}")
            continue

        h, w = img.shape[:2]

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                parsed = yolo_to_xyxy(line, w, h)
                if parsed is None:
                    continue

                cls_id, x1, y1, x2, y2 = parsed
                cls_name = CLASS_NAMES.get(cls_id, str(cls_id))

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(
                    img,
                    cls_name,
                    (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2
                )
        else:
            cv2.putText(
                img,
                "NO LABEL",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2
            )

        save_path = os.path.join(OUTPUT_DIR, image_name)
        cv2.imwrite(save_path, img)

        if (i + 1) % 20 == 0 or (i + 1) == len(image_paths):
            print(f"[{i+1}/{len(image_paths)}] 저장 완료")

    print(f"\n완료: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()