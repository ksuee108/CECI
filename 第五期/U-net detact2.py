import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import load_model
from keras_preprocessing.image import img_to_array, load_img
import os

# 1. 加载U-Net模型
model = load_model(r"C:\Users\user\Desktop\CECI\4th\Model\u-net.h5")

# 2. 辅助函数
def image_to_patches(image, patch_size):
    patches = []
    positions = []
    img_height, img_width = image.shape[:2]
    patch_height, patch_width = patch_size
    for y in range(0, img_height, patch_height):
        for x in range(0, img_width, patch_width):
            patch = image[y:y+patch_height, x:x+patch_width]
            if patch.shape[0] != patch_height or patch.shape[1] != patch_width:
                pad_height = patch_height - patch.shape[0]
                pad_width = patch_width - patch.shape[1]
                patch = np.pad(patch, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
            patches.append(patch)
            positions.append((y, x))

    return np.array(patches), positions

def patches_to_image(patches, positions, image_shape):
    reconstructed_image = np.zeros(image_shape, dtype=patches.dtype)
    patch_height, patch_width = patches.shape[1:3]
    for patch, (y, x) in zip(patches, positions):
        y_end = min(y + patch_height, image_shape[0])
        x_end = min(x + patch_width, image_shape[1])
        reconstructed_image[y:y_end, x:x_end] = patch[:y_end-y, :x_end-x]
    return reconstructed_image

# 文件夹路径
input_folder = r"C:\Users\user\Desktop\CECI\4th\edit2"
output_folder = r"C:\Users\user\Desktop\CECI\4th\edit2"
os.makedirs(output_folder, exist_ok=True)

patch_size = (512, 512)

# 3. 处理每个年份的图像
for year in range(2023, 2025):
    input_img_path = os.path.join(input_folder, f"{year}.png")
    if not os.path.exists(input_img_path):
        print(f"Input file not found: {input_img_path}")
        continue

    # 加载图像并切割为小块
    new_img = load_img(input_img_path)
    new_img_array = img_to_array(new_img) / 255.0
    original_height, original_width = new_img_array.shape[:2]
    patches, positions = image_to_patches(new_img_array, patch_size)
    print("check:", patches.shape)

    # 使用模型预测小块
    patch_inputs = np.array(patches)  # 修正输入形状
    predictions = model.predict_on_batch(patch_inputs)
    predictions = (predictions > 0.5).astype(np.uint8)  # 二值化

    # 重建预测的掩膜图像
    prediction_mask = patches_to_image(predictions, positions, (original_height, original_width, 1))

    # 后处理：形态学闭运算
    kernel_size = 7
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed_mask = cv2.morphologyEx(prediction_mask[:, :, 0], cv2.MORPH_CLOSE, kernel)

    # 可视化结果
    plt.imshow(closed_mask, cmap='gray')
    plt.title(f"Coastline Prediction for {year}")
    plt.axis('off')
    plt.show()

    # 保存处理后的掩膜图像
    output_image_path = os.path.join(output_folder, f"prediction_{year}.png")
    cv2.imwrite(output_image_path, (closed_mask * 255).astype(np.uint8))
    
    # 提取非零点的坐标并保存为 CSV
    coastline_coords = np.column_stack(np.where(closed_mask > 0))
    output_csv_path = os.path.join(output_folder, f"coastline_{year}2.csv")
    df_coords = pd.DataFrame(coastline_coords, columns=["Y", "X"])
    df_coords.to_csv(output_csv_path, index=False)

    print(f"Prediction saved: {output_image_path}")
    print(f"Coastline coordinates saved: {output_csv_path}")
