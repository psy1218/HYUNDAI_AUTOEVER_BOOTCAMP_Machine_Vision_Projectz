import os

def rename_images_in_class_folders(root_dir):
    """
    root_dir 아래의 각 클래스 폴더(예: 1_E, 2_N ...) 안 이미지 파일명을
    1_E_001.jpg 형태로 일괄 변경
    """

    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    for class_name in sorted(os.listdir(root_dir)):
        class_path = os.path.join(root_dir, class_name)

        if not os.path.isdir(class_path):
            continue

        # 폴더 안 이미지 파일만 수집
        image_files = [
            f for f in os.listdir(class_path)
            if os.path.isfile(os.path.join(class_path, f))
            and f.lower().endswith(valid_ext)
        ]

        image_files.sort()

        print(f"\n[{class_name}] 폴더 - {len(image_files)}개 파일 처리")

        # 이름 충돌 방지용: 먼저 임시 이름으로 변경
        temp_names = []
        for idx, old_name in enumerate(image_files, start=1):
            old_path = os.path.join(class_path, old_name)
            ext = os.path.splitext(old_name)[1].lower()
            temp_name = f"__temp__{idx:03d}{ext}"
            temp_path = os.path.join(class_path, temp_name)

            os.rename(old_path, temp_path)
            temp_names.append((temp_name, ext))

        # 최종 이름으로 변경
        for idx, (temp_name, ext) in enumerate(temp_names, start=1):
            temp_path = os.path.join(class_path, temp_name)
            new_name = f"{class_name}_{idx:03d}{ext}"
            new_path = os.path.join(class_path, new_name)

            os.rename(temp_path, new_path)
            print(f"{temp_name} -> {new_name}")

    print("\n모든 폴더의 파일명 변경 완료")


# 사용 예시
root_dir = "/home/psy1218/projects/1_pro/new_images/train"
rename_images_in_class_folders(root_dir)