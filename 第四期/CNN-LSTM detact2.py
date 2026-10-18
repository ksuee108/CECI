import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    roc_curve,
    auc
)
from skimage.transform import resize
from keras.models import load_model
from keras_preprocessing.image import img_to_array, load_img
import pandas as pd
import os

# 1. Load your LSTM model
model = load_model("C:\\Users\\user\\Desktop\\CECI\\4th\\Model\\CNN-LSTM.h5")

# 2. Define helper functions with overlap and blending
def image_to_patches(image, patch_size, overlap):
    patches = []
    positions = []
    img_height, img_width = image.shape[:2]
    patch_height, patch_width = patch_size
    step_y = patch_height - overlap
    step_x = patch_width - overlap
    for y in range(0, img_height - overlap, step_y):
        for x in range(0, img_width - overlap, step_x):
            patch = image[y:y+patch_height, x:x+patch_width]
            # If the patch is smaller than the patch size, pad it
            if patch.shape[0] != patch_height or patch.shape[1] != patch_width:
                pad_height = patch_height - patch.shape[0]
                pad_width = patch_width - patch.shape[1]
                patch = np.pad(
                    patch,
                    ((0, pad_height), (0, pad_width), (0, 0)),
                    mode='constant'
                )
            patches.append(patch)
            positions.append((y, x))
    return np.array(patches), positions

def patches_to_image(patches, positions, image_shape, overlap):
    """
    Reconstruct the original image from patches with overlapping and blending.

    Args:
        patches (numpy.ndarray): Array of patches with shape (num_patches, patch_height, patch_width, channels).
        positions (list): List of (y, x) positions where each patch is placed.
        image_shape (tuple): Shape of the original image (height, width, channels).
        overlap (int): Overlap size between patches.

    Returns:
        numpy.ndarray: Reconstructed image with shape `image_shape`.
    """
    reconstructed_image = np.zeros(image_shape, dtype=np.float32)  # Shape: (H, W, C)
    weight_mask = np.zeros(image_shape, dtype=np.float32)        # Shape: (H, W, C)
    patch_height, patch_width = patches.shape[1:3]

    for patch, (y, x) in zip(patches, positions):
        y_end = min(y + patch_height, image_shape[0])
        x_end = min(x + patch_width, image_shape[1])

        h = y_end - y
        w = x_end - x

        # 确保 patch 的维度为 (h, w, C)
        patch_crop = patch[:h, :w, 0]  # 取第一个通道，形状: (h, w)
        patch_crop = patch_crop[..., np.newaxis]  # 扩展为 (h, w, 1)

        # 将 patch 添加到重建图像
        reconstructed_image[y:y_end, x:x_end, 0] += patch_crop[:, :, 0]

        # 更新权重掩膜
        weight_mask[y:y_end, x:x_end, 0] += 1

    # 避免除以零
    weight_mask[weight_mask == 0] = 1

    # 平均重叠区域
    reconstructed_image /= weight_mask

    # 如果只处理单通道，可以移除多余的维度
    return reconstructed_image

# 3. Load and preprocess the input image
input_folder = r"C:\Users\user\Desktop\CECI\4th\edit2"
output_folder = r"C:\Users\user\Desktop\CECI\4th\Predictions\CNN-LSTM"
os.makedirs(output_folder, exist_ok=True)
patch_size = (128, 128)
for year in range(2024, 2030):
    input_img_path = os.path.join(input_folder, f"{year}.png")
    if not os.path.exists(input_img_path):
        print(f"Input file not found: {input_img_path}")
        continue
    new_img = load_img(input_img_path)

    new_img_array = img_to_array(new_img) / 255.0

    # Get original image dimensions
    original_height, original_width = new_img_array.shape[:2]

    # Define model input dimensions
    patch_size = (128, 128)  # Adjust based on your model's input size
    overlap = 32  # You can adjust this value based on your needs

    # Extract overlapping patches
    patches, positions = image_to_patches(new_img_array, patch_size, overlap)

    # 4. Predict on each patch
    timesteps = 3  # Ensure this matches your model's expected timesteps

    predicted_patches = []
    for i, patch in enumerate(patches):
        # Create a time sequence by repeating the same patch
        patch_sequence = np.array([patch] * timesteps)
        # Expand dimensions to match the model input shape
        patch_input = np.expand_dims(patch_sequence, axis=0)  # Shape: (1, timesteps, height, width, channels)
        # Predict
        prediction = model.predict(patch_input)
        predicted_patches.append(prediction[0])

    predicted_patches = np.array(predicted_patches)

    # Reconstruct the image with blending
    prediction_mask = patches_to_image(predicted_patches, positions, (original_height, original_width, 1), overlap)

    # Threshold the prediction mask to get binary mask
    prediction_mask_binary = (prediction_mask > 0.3).astype(np.uint8)

    # Apply morphological closing to the prediction mask
    kernel_size = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed_mask = cv2.morphologyEx(prediction_mask_binary[:, :, 0], cv2.MORPH_CLOSE, kernel)
    closed_mask = 1 - closed_mask
    # Visualize results
    plt.figure(figsize=(18, 6))

    plt.title('Closed Prediction Mask (Binary)')
    plt.imshow(closed_mask, cmap='gray')
    plt.axis('off')

    plt.show()
    # 保存处理后的掩膜图像
    output_image_path = os.path.join(output_folder, f"prediction_{year}.png")
    cv2.imwrite(output_image_path, (closed_mask * 255).astype(np.uint8))
    
    # 提取非零点的坐标并保存为 CSV
    coastline_coords = np.column_stack(np.where(closed_mask > 0))
    output_csv_path = os.path.join(output_folder, f"coastline_{year}.csv")
    df_coords = pd.DataFrame(coastline_coords, columns=["Y", "X"])
    df_coords.to_csv(output_csv_path, index=False)

    print(f"Prediction saved: {output_image_path}")
    print(f"Coastline coordinates saved: {output_csv_path}")
