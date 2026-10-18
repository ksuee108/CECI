import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import (Dense, LeakyReLU, BatchNormalization, Reshape, Flatten, Input, Concatenate, Conv2D, UpSampling2D)
from keras.optimizers import Adam
import os
from PIL import Image
from keras.layers import Conv2DTranspose
from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from sklearn.metrics import mean_squared_error
import pandas as pd
from scipy.linalg import sqrtm
from keras import mixed_precision
import tensorflow as tf
from scipy.stats import entropy
from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim
from datetime import datetime

def load_images_from_folder(folder_path, img_size):
    images = []
    
    # 获取文件夹中所有子目录或文件
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        print(f"Checking path: {item_path}")  # 打印当前检查的路径

        # 如果是文件夹，递归调用
        if item.endswith(".jpg") or item.endswith(".png"):
            try:
                img = Image.open(item_path).convert('L')  # 转为灰度图
                img = img.resize(img_size)  # 调整图像大小
                img = np.array(img)  # 转为 numpy 数组
                img = (img / 127.5) - 1.0  # 归一化到[-1, 1]
                images.append(img)
            except Exception as e:
                print(f"Error loading image {item_path}: {e}")  # 如果加载图像时发生错误，输出错误信息
        else:
            print(f"Skipping non-image file: {item_path}")  # 跳过非图像文件

    print(f"Total images loaded: {len(images)}")  # 调试信息
    return np.array(images)

def build_generator_with_condition(z_dim, year_dim):
    z_input = Input(shape=(z_dim,))
    year_input = Input(shape=(year_dim,))

    model_input = Concatenate()([z_input, year_input])
    # Starting spatial size (2,16) -> after 5 upsampling (64,512)
    x = Dense(2 * 32 * 256)(model_input)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Reshape((2, 32, 256))(x)

    # Upsample blocks: UpSampling2D + Conv2D to avoid checkerboard artifacts
    for filters in [128, 64, 32, 16]:
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2DTranspose(filters, kernel_size=3, padding='same')(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)

    """
    x = UpSampling2D(size=(1, 2))(x)
    x = Conv2DTranspose(8, kernel_size=3, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)
    """

    # Additional conv layers for refinement
    x = Conv2DTranspose(4, kernel_size=3, padding='same')(x)
    x = Conv2DTranspose(4, kernel_size=3, padding='same')(x)
    
    for filters in [16, 32, 64, 128]:
        x = Conv2DTranspose(filters, kernel_size=3, padding='same')(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)

    # Final output layer for (512, 512, 1)
    img_output = Conv2DTranspose(1, kernel_size=3, strides=2, padding='same', activation='tanh')(x)  # (512, 512, 1) # strides=2 壓縮
    print("img_output", img_output.shape)
    return Model([z_input, year_input], img_output)

def build_discriminator(img_shape):
    inp = Input(shape=img_shape)

    x = Conv2D(64, 4, strides=2, padding='same')(inp)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(128, 4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(256, 4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Flatten()(x)
    out = Dense(1, activation='sigmoid')(x)

    return Model(inp, out)

def build_gan(generator, discriminator, year_dim):
    discriminator.trainable = False

    # 定义输入：噪声向量 z 和年份条件 year
    z = Input(shape=(z_dim,))
    year = Input(shape=(year_dim,))  # 新增年份输入
    
    # 使用两个输入调用生成器
    img = generator([z, year])  # 生成器需要两个输入: z 和 year
    
    # 判别器输出
    validity = discriminator(img)

    # 构建 GAN 模型，输入是 z 和 year，输出是判别器的概率
    model = Model([z, year], validity)
    return model

"""
def train(epochs, batch_size=128, sample_interval=50):
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        # 训练判别器
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_images = X_train[idx]

        # 生成假图像
        noise = np.random.randn(batch_size, z_dim)
        
        # 假设我们在这里使用的年份是从 0 到 11 之间的整数（对应12个年份）
        # 你可以根据你的数据设定年份的条件
        year_conditions = np.random.randint(0, 12, batch_size)  # 随机选择年份作为条件
        year_conditions = np.eye(12)[year_conditions]  # 使用 one-hot 编码生成 (batch_size, 12) 的数组

        # 将噪声和年份条件拼接为生成器的输入
        fake_images = generator.predict([noise, year_conditions])

        # 训练判别器：真实图像为1，假图像为0
        d_loss_real = discriminator.train_on_batch(real_images, valid)
        d_loss_fake = discriminator.train_on_batch(fake_images, fake)

        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # 训练生成器（通过 GAN 模型）：让生成器生成的图像被判别器认为是真实的
        g_loss = gan.train_on_batch([noise, year_conditions], valid)

        # 每隔一定间隔打印损失值并生成图像
        if epoch % sample_interval == 0:
            print(f"{epoch} [D loss: {d_loss[0]} | D accuracy: {100*d_loss[1]}] [G loss: {g_loss}]")
            sample_images(epoch)
"""
def sample_images(epoch, year_dim, output_size=(512, 512)):
    # 生成噪声和年份条件
    noise = np.random.randn(1, z_dim)
    year_conditions = np.random.randint(0, year_dim, 1)
    year_conditions = np.eye(year_dim)[year_conditions]

    # 使用生成器生成图像
    gen_image = generator.predict([noise, year_conditions])[0]  # 提取生成的单张图像


    # 去归一化至 [0, 1]，然后转为 [0, 255]
    gen_image = 0.5 * gen_image + 0.5
    gen_image = (gen_image * 255).astype(np.uint8)

    # 动态创建目录
    if not os.path.exists(f"gent\\bin\\images\\test2\\{run_time}"):
        os.makedirs(f"gent\\bin\\images\\test2\\{run_time}")
    save_dir = os.path.join(f"gent\\bin\\images\\test2\\{run_time}", str(year_dim))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"sample_{epoch}.png")
    Image.fromarray(gen_image[:, :, 0]).save(save_path)
    print(f"Sample image saved to {save_path}")

    """

    # 调整图像大小为目标尺寸
    gen_image_resized = Image.fromarray(gen_image[:, :, 0]).resize(output_size, Image.LANCZOS)

    # 保存生成的图像
    save_path = f"C:\\Users\\user\\Desktop\\CECI\\4th\\images\\sample_{epoch}.png"
    gen_image_resized.save(save_path)  # 保存调整大小后的图像
    print(f"Single sample image saved to {save_path}")
    """

# 转换为 InceptionV3 输入大小
def preprocess_images(images):
    # 灰度图像扩展为 RGB，并确保数据类型为 uint8
    rgb_images = np.array([np.repeat(img, 3, axis=-1) for img in images])  # 扩展通道
    rgb_images = np.clip(0.5 * rgb_images + 0.5, 0, 1)  # 归一化到 [0, 1]
    rgb_images = (rgb_images * 255).astype(np.uint8)  # 转为 uint8 类型
    resized_images = np.array([np.array(Image.fromarray(img).resize((299, 299))) for img in rgb_images])
    return preprocess_input(resized_images)

# 计算 Inception Score (IS)
def calculate_inception_score(images, inception_model, splits=10):
    """
    计算给定图像集合的 Inception Score (IS)。

    参数：
    - images: np.ndarray
        输入图像数组，需经过预处理（例如调整尺寸和归一化）。
    - inception_model: keras.Model 或等效模型
        预训练的 Inception 模型，用于生成预测结果。
    - splits: int, 可选 (默认值=10)
        将数据分为若干组，以提高分数计算的稳定性。

    返回：
    - is_score: float
        计算得到的 Inception Score。
    """
    # 第一步：对输入图像进行预处理
    processed_images = preprocess_images(images)  # 需要自行定义 preprocess_images 函数
    
    # 第二步：通过 Inception 模型预测概率分布
    preds = inception_model.predict(processed_images, batch_size=32, verbose=1)
    
    # 第三步：将预测结果按 splits 分组
    split_scores = []
    n_images = len(processed_images)
    for i in range(splits):
        # 获取分组数据
        part = preds[i * (n_images // splits): (i + 1) * (n_images // splits), :]
        
        # 第四步：计算每组的 KL 散度和得分
        p_y = np.mean(part, axis=0)  # 每组的边缘分布
        kl_divergences = [entropy(part[j], p_y) for j in range(part.shape[0])]  # KL 散度
        split_score = np.exp(np.mean(kl_divergences))  # 计算每组得分
        split_scores.append(split_score)
    
    # 第五步：返回所有分组得分的平均值
    is_score = np.mean(split_scores)
    return is_score

# 计算 Frechet Inception Distance (FID)
def calculate_fid(real_images, fake_images):
    real_features = inception_model.predict(preprocess_images(real_images), batch_size=32)
    fake_features = inception_model.predict(preprocess_images(fake_images), batch_size=32)

    # Compute mean and covariance
    mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)

    # Compute the squared difference of means
    diff = mu1 - mu2
    diff_squared = diff.dot(diff)

    # Compute the covariance mean
    covmean = sqrtm(sigma1.dot(sigma2) + 1e-6 * np.eye(sigma1.shape[0]))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Calculate FID score
    fid_score = diff_squared + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid_score

# 更新训练函数，增加评估指标和 CSV 记录
def train_with_metrics(epochs, batch_size=64, sample_interval=50, year_dim=12):
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    best_fid = float('inf')
    #results = []  # 存储评估结果

    for epoch in range(epochs):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_images = X_train[idx]
        
        noise = np.random.randn(batch_size, z_dim)
        year_conditions = np.random.randint(0, year_dim, batch_size)
        year_conditions = np.eye(year_dim)[year_conditions]
        fake_images = generator.predict([noise, year_conditions])

        d_loss_real = discriminator.train_on_batch(real_images, valid)
        d_loss_fake = discriminator.train_on_batch(fake_images, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        g_loss = gan.train_on_batch([noise, year_conditions], valid)

        if epoch % sample_interval == 0:
            print(f"{epoch} [D loss: {d_loss[0]} | D accuracy: {100*d_loss[1]}] [G loss: {g_loss}]")
            # 每隔 sample_interval 计算 IS、FID 和 PSNR
            fake_samples = generator.predict([np.random.randn(100, z_dim), np.eye(year_dim)[np.random.randint(0, year_dim, 100)]])
            print("X_train:",X_train.shape, "Fake_samples:",fake_samples.shape)
            fid_score = calculate_fid(X_train, fake_samples)
            is_score = calculate_inception_score(fake_samples,inception_model)

            print(f"Epoch {epoch} | FID: {fid_score:.2f} | IS: {is_score:.2f}")
            
            sample_images(epoch, year_dim)
            
            # 保存结果到 CSV
            results = { "epoch": epoch, 
                        "FID": fid_score, 
                        "IS": is_score, 
                        "D loss": d_loss[0], 
                        'D accuracy': 100*d_loss[1], 
                        'G loss': g_loss}

            # 保存 CSV 文件
            results_df = pd.DataFrame([results])
            
            csv_path = os.path.join(base_output_dir, f"metrics_results_{year_dim}.csv")
            results_df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
            #results_df.to_csv(csv_path, index=False)
            print(f"Metrics saved to {csv_path}")
            if not np.isnan(fid_score) and fid_score < best_fid:
                best_fid = fid_score
                best_generator_path = os.path.join(base_output_dir, f"generator_year_dim_{year_dim}_best_{epoch}.h5")
                generator.save(best_generator_path)
                print(f"✅ 保存最佳 generator (FID: {fid_score:.4f}) 到 {best_generator_path}")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("已设置 GPU 内存动态增长。")
    except RuntimeError as e:
        print(e)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# 加载预训练的 InceptionV3 模型（用于 IS 和 FID 计算）
inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
base_output_dir = os.path.join("gent", "bin", "images", "test2", run_time)
os.makedirs(base_output_dir, exist_ok=True)
# 假设 input_img_path 目录中包含 12 年的数据
input_img_path = r"D:\desk\CECI\coastline_CEC V\new\main file\data\north\Finish\segmentation\balanced"
img_size = (1024, 64)

# 加载所有图像（包括12年的数据）
X_train = load_images_from_folder(input_img_path, img_size)
print(f"X_train.shape: {X_train.shape}")  # Debugging print
X_train = X_train.reshape(X_train.shape[0], 64, 1024, 1)

z_dim = 128  # 噪声向量的维度
img_shape = (64, 1024, 1)  # 图像的形状
optimizer = Adam(0.0002, 0.5)

discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

for initial_year_dim in range(12, 17):  # 动态增加年份维度
    generator = build_generator_with_condition(z_dim, initial_year_dim)
    gan = build_gan(generator, discriminator, initial_year_dim)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    train_with_metrics(epochs=20001, batch_size=64, sample_interval=100, year_dim=initial_year_dim)
    generator.save(os.path.join(base_output_dir, f"generator_year_dim_{initial_year_dim}.h5"))
    discriminator.save(os.path.join(base_output_dir, f"discriminator_year_dim_{initial_year_dim}.h5"))
    gan.save(os.path.join(base_output_dir, f"gan_full_year_dim_{initial_year_dim}.h5"))
    save_path = f"base_output_dir\\{run_time}"

    for y in range(initial_year_dim):
        sample_images(
            epoch='final',
            year_dim=initial_year_dim,
        )
    input("Tap any key to continue after reviewing the new GAN image...")