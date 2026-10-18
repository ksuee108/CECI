import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import load_model

model = load_model("C:\\Users\\user\\Desktop\\CECI\\4th\\Model\\LSTM.h5")
# 定义图像尺寸和像素数
IMG_HEIGHT = 128
IMG_WIDTH = 128
NUM_PIXELS = IMG_HEIGHT * IMG_WIDTH

# 预处理图像函数
def preprocess_image(image_path):
    # 加载图像并转换为灰度图像
    img = Image.open(image_path).convert("L")
    # 调整图像大小
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    # 转换为 NumPy 数组并标准化到 [0, 1]
    img_array = np.array(img).astype(np.float32) / 255.0
    # 展平成一维数组
    img_flat = img_array.reshape(1, NUM_PIXELS)
    return img_flat

input_folder = r"C:\Users\user\Desktop\CECI\4th\edit2"
output_folder = r"C:\Users\user\Desktop\CECI\4th\Predictions"
os.makedirs(output_folder, exist_ok=True)

patch_size = (512, 512)

# 3. 处理每个年份的图像
for year in range(2024, 2030):
    input_img_path = os.path.join(input_folder, f"{year}.png")
    if not os.path.exists(input_img_path):
        print(f"Input file not found: {input_img_path}")
        continue


    # 预处理图像
    image_flat = preprocess_image(input_img_path)

    # 使用模型进行预测
    predicted_mask_flat = model.predict(image_flat)

    # 后处理预测结果
    def postprocess_mask(predicted_mask_flat):
        # 将预测结果从一维展开为二维
        predicted_mask = predicted_mask_flat.reshape(IMG_HEIGHT, IMG_WIDTH)
        # 应用阈值，将概率值转换为二值掩码
        predicted_mask_binary = (predicted_mask > 0.4).astype(np.uint8)
        return predicted_mask_binary

    # 获取二值化预测掩码
    predicted_mask = postprocess_mask(predicted_mask_flat)

    # 显示原始图像和预测的掩码
    # 加载原始图像用于显示

    plt.figure(figsize=(12, 6))

    plt.imshow(predicted_mask, cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')

    plt.show()