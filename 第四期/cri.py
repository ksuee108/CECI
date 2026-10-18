import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import load_model
from keras_preprocessing.image import img_to_array, load_img
import os
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
)
from skimage.transform import resize

# Mean Absolute Percentage Error
def mean_absolute_percentage_error(y_true, y_pred):
    """
    计算平均绝对百分比误差 (MAPE)

    Parameters:
        y_true (array-like): 实际值
        y_pred (array-like): 预测值

    Returns:
        float: MAPE 值
    """
    # 展平为 1D 数组
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # 过滤掉 y_true 中为 0 的元素
    non_zero_indices = y_true != 0
    y_true_non_zero = y_true[non_zero_indices]
    y_pred_non_zero = y_pred[non_zero_indices]
    
    # 如果过滤后没有数据，返回 NaN
    if len(y_true_non_zero) == 0:
        print("Warning: No non-zero elements in y_true. MAPE is undefined.")
        return np.nan
    
    # 计算 MAPE
    mape = np.mean(np.abs((y_true_non_zero - y_pred_non_zero) / y_true_non_zero)) * 100
    return mape

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

all_results = []
models = ['U-net', 'CNN', 'CNN-LSTM']#
# 文件夹路径
input_folder = r"C:\Users\user\Desktop\CECI\4th\edit2"
for model_name in models:  # Use model_name to preserve the string
    # Use model_name for naming
    model_path = f"C:\\Users\\user\\Desktop\\CECI\\4th\\Model\\{model_name}.h5"
    output_folder = f"C:\\Users\\user\\Desktop\\CECI\\4th\\Predictions\\{model_name}"
    os.makedirs(output_folder, exist_ok=True)

    patch_size = (512, 512)

    # 3. 处理每个年份的图像
    for year in range(2024, 2030):
        print(year)
        input_img_path = os.path.join(input_folder, f"{year}.png")
        if not os.path.exists(input_img_path):
            print(f"Input file not found: {input_img_path}")
            continue

        # 加载图像并切割为小块
        new_img = load_img(input_img_path)
        new_img_array = img_to_array(new_img) / 255.0
        print(new_img_array.shape)
        output_img_path = os.path.join(output_folder, f"prediction_{year}.png")
        if not os.path.exists(output_img_path):
            print(f"Input file not found: {output_img_path}")
            continue

        # 加载图像并切割为小块
        predicted_img = load_img(output_img_path)
        predicted_img_array = img_to_array(predicted_img) / 255.0
        print(predicted_img_array.shape)

        """# ----------------------- 评估指标计算部分 -----------------------
        mask_img_array = new_img_array[:, :, 0]  # Keep only one channel
        predicted_img_array = predicted_img_array[:, :, 0]  # 取第一通道

        # Ensure the ground truth mask size matches the prediction mask
        if mask_img_array.shape != predicted_img_array.shape[:2]:
            mask_img_array_resized = resize(mask_img_array, predicted_img_array.shape[:2], preserve_range=True)
            mask_img_array_resized = mask_img_array_resized.astype(np.float32)  # Ensure float type for consistency
        else:
            mask_img_array_resized = mask_img_array

        # Binarize the true values
        true_values = (mask_img_array_resized > 0.5).astype(np.float32)
        predicted_img_array = (predicted_img_array > 0.5).astype(np.float32)

        print("true_values shape:", true_values.shape)
        print("predicted_img_array shape:", predicted_img_array.shape)
        """
        
        new_img_flat = new_img_array.flatten()
        predicted_img_flat = predicted_img_array.flatten()
        mae = mean_absolute_error(new_img_flat, predicted_img_flat)

        # Root Mean Squared Error
        rmse = np.sqrt(mean_squared_error(new_img_flat, predicted_img_flat))

        # Intersection over Union (IoU)
        intersection = np.logical_and(new_img_flat > 0.5, predicted_img_flat > 0.5).sum()
        union = np.logical_or(new_img_flat > 0.5, predicted_img_flat > 0.5).sum()
        iou_score = intersection / union if union != 0 else 1

        # Accuracy
        accuracy = accuracy_score((new_img_flat > 0.5).astype(int), (predicted_img_flat > 0.5).astype(int))

        # F1 Score
        f1 = f1_score((new_img_flat > 0.5).astype(int), (predicted_img_flat > 0.5).astype(int), zero_division=0, average='micro')

        # Dice Coefficient
        dice_score = 2 * intersection / (new_img_flat.sum() + predicted_img_flat.sum()) if (new_img_flat.sum() + predicted_img_flat.sum()) != 0 else 1

        # Precision and Recall
        precision = precision_score((new_img_flat > 0.5).astype(int), (predicted_img_flat > 0.5).astype(int), zero_division=0, average='micro')
        recall = recall_score((new_img_flat > 0.5).astype(int), (predicted_img_flat > 0.5).astype(int), zero_division=0, average='micro')
        
        mape = mean_absolute_percentage_error(new_img_flat, predicted_img_flat)
        
        # Save results
        all_results.append({
            "Model": model_name,
            "Year": year,
            "MAPE": mape,
            "MAE": mae,
            "RMSE": rmse,
            "IoU": iou_score,
            "Accuracy": accuracy,
            "F1 Score": f1,
            "Dice Coefficient": dice_score,
            "Precision": precision,
            "Recall": recall,
        })

# Save all results to a single CSV file
results_df = pd.DataFrame(all_results)
results_csv_path = r"C:\Users\user\Desktop\CECI\4th\Evaluation_Results.csv"
results_df.to_csv(results_csv_path, index=False)
print(f"Results saved to {results_csv_path}")