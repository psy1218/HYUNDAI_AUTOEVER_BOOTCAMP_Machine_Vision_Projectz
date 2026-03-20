import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_class_distribution(root_dir):
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    results = []

    for class_name in sorted(os.listdir(root_dir)):
        class_path = os.path.join(root_dir, class_name)

        if not os.path.isdir(class_path):
            continue

        image_count = sum(
            1 for f in os.listdir(class_path)
            if os.path.isfile(os.path.join(class_path, f))
            and f.lower().endswith(valid_ext)
        )

        results.append({
            'class_name': class_name,
            'image_count': image_count
        })

    df = pd.DataFrame(results)

    if df.empty:
        print("표시할 데이터가 없습니다.")
        return None

    plt.figure(figsize=(14, 6))
    plt.bar(df['class_name'], df['image_count'])
    plt.title('클래스별 이미지 개수 분포')
    plt.xlabel('클래스')
    plt.ylabel('이미지 수')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return df


# 사용 예시
train_root = "/home/psy1218/projects/1_pro/images/train"
df_plot = plot_class_distribution(train_root)