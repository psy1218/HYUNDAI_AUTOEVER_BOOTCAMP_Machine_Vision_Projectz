import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from torchvision import transforms
import os

def show_resize_comparison(image_paths):
    if len(image_paths) != 3:
        raise ValueError("이미지 경로는 반드시 3개여야 합니다.")

    # 전처리 정의
    transform_resize_224 = transforms.Compose([
        transforms.Resize((224, 224))
    ])

    transform_resize_384_256 = transforms.Compose([
        transforms.Resize((384, 256))   # 세로형
    ])

    transform_resize_384_288 = transforms.Compose([
        transforms.Resize((384, 288))   # 세로형
    ])

    original_images = []
    file_names = []

    for path in image_paths:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        original_images.append(img)
        file_names.append(os.path.basename(path))

    resized_224_images = [transform_resize_224(img) for img in original_images]
    resized_384_256_images = [transform_resize_384_256(img) for img in original_images]
    resized_384_288_images = [transform_resize_384_288(img) for img in original_images]

    fig, axes = plt.subplots(4, 3, figsize=(12, 20))

    row_titles = [
        "Original",
        "1) Resize((224,224))",
        "2) Resize((384,256))",
        "3) Resize((384,288))"
    ]

    image_groups = [
        original_images,
        resized_224_images,
        resized_384_256_images,
        resized_384_288_images
    ]

    for row in range(4):
        for col in range(3):
            axes[row, col].imshow(image_groups[row][col])
            axes[row, col].axis("off")

            if row == 0:
                axes[row, col].set_title(file_names[col], fontsize=11)

        axes[row, 0].set_ylabel(row_titles[row], fontsize=12, rotation=90, labelpad=25)

    plt.tight_layout()
    plt.show()


image_paths = [
    "/home/psy1218/projects/1_pro/images/img1.jpg",
    "/home/psy1218/projects/1_pro/images/img2.jpg",
    "/home/psy1218/projects/1_pro/images/img3.jpg"
]

show_resize_comparison(image_paths)