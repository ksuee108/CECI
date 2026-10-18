import numpy as np
from keras.models import Sequential, Model, load_model
from keras.layers import (Dense, LeakyReLU, BatchNormalization, Reshape, Flatten, Input, Concatenate, 
                          Conv2D, UpSampling2D, Add, GlobalAveragePooling2D, )
from tensorflow_addons.layers import SpectralNormalization
from keras.optimizers import Adam
import os
from PIL import Image
import pandas as pd
from image_metrics import Metrics
import tensorflow as tf
from keras.applications import InceptionV3
from datetime import datetime
from keras import mixed_precision
import cv2
from concurrent.futures import ProcessPoolExecutor

def load_and_process_image(path, img_size):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)
    img = (img / 127.5) - 1.0
    return img

def process_image(args):
    path, img_size = args
    return load_and_process_image(path, img_size)

def load_images_from_folder_fast(
    root_dir,
    img_size,
    max_images_per_folder=50,
    num_workers=8,
    save_path="train_images.npy"
):
    image_paths = []

    # 1️⃣ 先蒐集所有路徑（I/O 與 compute 分離）
    for folder, _, files in os.walk(root_dir):
        imgs = [
            os.path.join(folder, f)
            for f in files
            if f.lower().endswith((".jpg", ".png"))
        ]#[:max_images_per_folder]
        image_paths.extend(imgs)
        print(f"Found {len(imgs)} images in {folder}")

    print(f"Total images to load: {len(image_paths)}")

    # 2️⃣ 使用 ProcessPool（真正平行）
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            executor.map(
                process_image,
                [(p, img_size) for p in image_paths]
            )
        )

    images = np.asarray([r for r in results if r is not None], dtype=np.float32)

    # 3️⃣ 一次性存檔（關鍵）
    np.save(save_path, images)

    print(f"Saved {images.shape[0]} images to {save_path}")
    return images

def sample_images(generator, epoch, year_dim, output_size=(512, 64)):
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
    if not os.path.exists(f"gent\\segmentation\\images\\test2\\{run_time}"):
        os.makedirs(f"gent\\segmentation\\images\\test2\\{run_time}")
    save_dir = os.path.join(f"gent\\segmentation\\images\\test2\\{run_time}", str(year_dim))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"sample_{epoch}.png")
    Image.fromarray(gen_image) .save(save_path)
    print(f"Sample image saved to {save_path}")

class GAN:
    def ResUpBlock(x, filters, kernel_size=3):
        """Residual Upsampling Block"""
        # shortcut
        sc = UpSampling2D()(x)
        sc = Conv2D(filters, 1, padding="same")(sc)

        # main path
        x = UpSampling2D()(x)
        x = Conv2D(filters, kernel_size, padding="same")(x)
        x = LeakyReLU(0.2)(x)

        x = Conv2D(filters, kernel_size, padding="same")(x)
        x = LeakyReLU(0.2)(x)
        
        x = Conv2D(filters, kernel_size, padding="same")(x)
        x = LeakyReLU(0.2)(x)
        
        x = Add()([x, sc])
        x = LeakyReLU(0.2)(x)
        return x
    
    def ResDownBlock(x, filters):
        shortcut = x

        x = Conv2D(filters, 3, strides=2, padding="same")(x)
        x = LeakyReLU(0.2)(x)

        x = Conv2D(filters, 3, padding="same")(x)
        x = Conv2D(filters, 3, padding="same")(x)
        x = Conv2D(filters, 3, padding="same")(x)

        shortcut = Conv2D(filters, 1, strides=2, padding="same")(shortcut)

        x = tf.keras.layers.Add()([x, shortcut])
        x = LeakyReLU(0.2)(x)

        return x

    def build_generator(z_dim=128, cond_dim=None):
        if cond_dim is not None:
            z = Input(shape=(z_dim,))
            c = Input(shape=(cond_dim,))
            x = tf.keras.layers.Concatenate()([z, c])
            inputs = [z, c]
        else:
            x = Input(shape=(z_dim,))
            inputs = x

        # ---------------------------
        # Latent projection
        # ---------------------------
        """x = Dense(1024)(x)
        x = LeakyReLU(0.2)(x)

        x = Dense(512)(x)
        x = LeakyReLU(0.2)(x)"""

        x = Dense(4 * 32 * 64)(x)
        x = LeakyReLU(0.2)(x)
        x = Reshape((4, 32, 64))(x)

        # ---------------------------
        # Progressive upsampling
        # ---------------------------
        x = Conv2D(128, 3, padding="same")(x)
        for filters in [32, 32, 16, 16, 10]:
            x = GAN.ResUpBlock(x, filters)
            x = Conv2D(filters, 3, padding="same")(x)
            x = Conv2D(filters*2, 3, padding="same")(x)
        #x = GlobalAveragePooling2D()(x)

        # ---------------------------
        # Refinement
        # ---------------------------
        x = Conv2D(8, 3, padding="same")(x)
        x = LeakyReLU(0.2)(x)

        x = Conv2D(8, 3, padding="same")(x)
        x = LeakyReLU(0.2)(x)

        x = Conv2D(8, 3, padding="same")(x)
        x = LeakyReLU(0.2)(x)

        # ---------------------------
        # Output
        # ---------------------------
        out = Conv2D(3, 3, padding="same", activation="sigmoid", dtype="float32")(x)

        return Model(inputs, out, name="Generator")

    def build_discriminator(img_shape, cond_dim=None):
        img = Input(shape=img_shape)

        x = SpectralNormalization(Conv2D(8, 3, padding="same"))(img)
        x = LeakyReLU(0.2)(x)

        # 對稱於 Generator 的 ResUpBlock
        for filters in [8, 8, 10, 16, 16, 32, 64]:
            x = GAN.ResDownBlock(x, filters)
        
        x = Conv2D(4, 3, padding="same")(x)

        x = GlobalAveragePooling2D()(x)

        # -------------------------
        # unconditional output
        # -------------------------
        out = SpectralNormalization(Dense(1))(x)

        # -------------------------
        # conditional projection
        # -------------------------
        if cond_dim is not None:
            c = Input(shape=(cond_dim,))
            channels = x.shape[-1] if x.shape[-1] is not None else 256
            embed = Dense(channels, use_bias=False)(c)
            proj = tf.reduce_sum(x * embed, axis=1, keepdims=True)
            out = out + proj
            inputs = [img, c]
        else:
            inputs = img

        return Model(inputs, out, name="Discriminator")

    def build_gan(generator, discriminator, year_dim):
        discriminator.trainable = False

        # 定义输入：噪声向量 z 和年份条件 year
        z = Input(shape=(z_dim,))
        year = Input(shape=(year_dim,))  # 新增年份输入
        
        # 使用两个输入调用生成器
        img = generator([z, year])  # 生成器需要两个输入: z 和 year
        
        # 判别器输出
        validity = discriminator([img, year])

        # 构建 GAN 模型，输入是 z 和 year，输出是判别器的概率
        model = Model([z, year], validity)
        return model

# 更新训练函数，增加评估指标和 CSV 记录
def train_with_metrics(
    generator,
    discriminator,
    epochs,
    batch_size=32,
    sample_interval=50,
    year_dim=12,
    base_output_dir="output"
    ):

    valid = np.ones((batch_size, 1)) * 0.9 + 0.05 * np.random.rand(batch_size, 1)  # label smoothing
    fake = 0.05 * np.random.rand(batch_size, 1)
    
    output_csv = os.path.join(base_output_dir, f"metrics_results_{year_dim}_V.csv")
    best_fid = float('inf')

    # 初始化 CSV
    if not os.path.exists(output_csv):
        pd.DataFrame(columns=["epoch", "d_loss", "g_loss", "FID", "PSNR", "SSIM"]).to_csv(output_csv, index=False)#, "IS"

    for epoch in range(epochs):
        # ---------------------
        #  Train Discriminator
        # ---------------------
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_images = X_train[idx]

        noise = np.random.randn(batch_size, z_dim)
        year_conditions = np.random.randint(0, year_dim, batch_size)
        year_conditions = np.eye(year_dim)[year_conditions]

        fake_images = generator.predict([noise, year_conditions])

        d_loss_real = discriminator.train_on_batch([real_images, year_conditions], valid)
        d_loss_fake = discriminator.train_on_batch([fake_images, year_conditions], fake)

        d_loss_val = 0.5 * (d_loss_real + d_loss_fake)

        # ---------------------
        #  Train Generator
        # ---------------------
        alpha, beta = 10.0, 5.0
        with tf.GradientTape() as tape:
            fake_images = generator([noise, year_conditions], training=True)
            validity = discriminator([fake_images, year_conditions], training=False)

            # 強制轉成 float32 避免混合精度衝突
            fake_images_f32 = tf.cast(fake_images, tf.float32)
            real_images_f32 = tf.cast(real_images, tf.float32)
            validity_f32 = tf.cast(validity, tf.float32)

            adv_loss = tf.keras.losses.hinge(valid, validity_f32)
            #adv_loss = tf.keras.losses.binary_crossentropy(valid, validity_f32)
            adv_loss = tf.reduce_mean(adv_loss)

            pixel_loss = tf.reduce_mean(tf.abs(fake_images_f32 - real_images_f32))
            ssim_loss  = 1.0 - tf.reduce_mean(tf.image.ssim(fake_images_f32, real_images_f32, max_val=2.0))
    
            g_loss = adv_loss + alpha * pixel_loss + beta * ssim_loss

        grads = tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))
        g_loss_val = float(tf.reduce_mean(g_loss))

        # ---------------------
        #  Logging & Sample
        # ---------------------
        if epoch % sample_interval == 0:
            print(f"{epoch} [D loss: {d_loss_val:.4f} |[G loss: {g_loss_val:.4f}]")

            # 生成測試樣本
            N_FID = 512
            test_noise = np.random.randn(N_FID, z_dim)
            test_years = np.random.randint(0, year_dim, N_FID)
            test_years = np.eye(year_dim)[test_years]
            gen_samples = generator.predict([test_noise, test_years], verbose=0)

            #with tf.device('/CPU:0'):
            #is_score = metrics.calculate_inception_score(gen_samples)
            fid_score = metrics.calculate_fid(fid_real_ref, gen_samples)

            n = min(len(real_images), len(gen_samples))
            psnr_score = metrics.calculate_psnr(real_images[:n], gen_samples[:n])
            ssim_score = metrics.calculate_average_ssim(real_images[:n], gen_samples[:n])

            # 儲存到 CSV
            new_row = pd.DataFrame([{
                "epoch": epoch,
                "d_loss": d_loss_val,
                "g_loss": g_loss_val,
                #"IS": is_score,
                "FID": fid_score,
                "PSNR": psnr_score,
                "SSIM": ssim_score
            }])
            new_row.to_csv(output_csv, mode="a", header=False, index=False)

            # 存樣本圖片
            sample_images(generator, epoch, year_dim)

        if not np.isnan(fid_score) and fid_score < best_fid:
            best_fid = fid_score
            best_generator_path = os.path.join(base_output_dir, f"generator_year_dim_{year_dim}_best_model.h5")
            generator.save(best_generator_path)
            print(f"✅ 保存最佳 generator (FID: {fid_score:.4f}) 到 {best_generator_path}")
        generator_path = os.path.join(base_output_dir, f"generator_year_dim_{year_dim}_last_model.h5")
        generator.save(generator_path)
        tf.keras.backend.clear_session()

"""mixed_precision.set_global_policy("mixed_float16")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("已设置 GPU 内存动态增长。")
    except RuntimeError as e:
        print(e)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'"""

def main():
    global d_optimizer, g_optimizer, fid_real_ref
    img_shape = (128, 1024, 3)  # 修改成生成器實際輸出
    d_optimizer = Adam(0.0002, 0.5)
    g_optimizer = Adam(0.0002, 0.5)

    for initial_year_dim in range(12, 17):  # 动态增加年份维度
        FID_REF_SIZE = 512
        fid_real_ref = X_train[
            np.random.choice(len(X_train), FID_REF_SIZE, replace=False)
        ]

        discriminator = GAN.build_discriminator(img_shape, cond_dim=initial_year_dim)
        discriminator.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=d_optimizer
    )

        generator = GAN.build_generator(z_dim, initial_year_dim)
        gan = GAN.build_gan(generator, discriminator, initial_year_dim)

        gan.compile(loss='binary_crossentropy', optimizer=g_optimizer)

        base_output_dir = os.path.join("gent", "segmentation", "images", "test2", run_time)
        os.makedirs(base_output_dir, exist_ok=True)
        
        train_with_metrics(generator, discriminator, epochs=2001, batch_size=8, sample_interval=2, year_dim=initial_year_dim, base_output_dir=base_output_dir)

        # 生成并保存一张样本图像以供检查
        sample_images(generator, "final", initial_year_dim)

        generator.save(os.path.join(base_output_dir, f"generator_year_dim_{initial_year_dim}.h5"))
        discriminator.save(os.path.join(base_output_dir, f"discriminator_year_dim_{initial_year_dim}.h5"))
        gan.save(os.path.join(base_output_dir, f"gan_full_year_dim_{initial_year_dim}.h5"))
        input("Tap any key to continue after reviewing the new GAN image...")

if __name__ == "__main__":
    z_dim = 256  # 噪声向量的维度
    inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
    metrics = Metrics(inception_model=inception_model)
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 假设 input_img_path 目录中包含 12 年的数据
    input_img_path = "new\\main file\\data\\north\\Finish\\segmentation\\balanced8"
    img_size = (1024, 128)
    # 加载所有图像（包括12年的数据）
    #X_train = load_images_from_folder_fast(input_img_path, img_size)
    X_train = np.load("train_images.npy", mmap_mode="r")
    #X_train = np.load("train_images.npy").astype(np.float32)
    print(f"X_train.shape: {X_train.shape}")  # Debugging print
    main()