import cv2
import numpy as np
import os
import shutil
import glob


def hsv_augmentation(image_path, output_dir, label_dir, aug_count=2):
    """生成HSV增强图像并复制标注文件"""
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误: 无法读取图像 {image_path}")
        return

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 获取基础文件名
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 生成多版本增强
    for i in range(aug_count):
        # 创建随机HSV变换参数
        h_shift = np.random.randint(-120, 120)  # 色相偏移
        s_scale = np.random.uniform(0.8, 1.2)  # 饱和度缩放
        v_scale = np.random.uniform(0.8, 1.2)  # 明度缩放

        # 应用变换
        aug_hsv = img_hsv.copy()
        aug_hsv[:, :, 0] = (aug_hsv[:, :, 0] + h_shift) % 180  # 色相循环
        aug_hsv[:, :, 1] = np.clip(aug_hsv[:, :, 1] * s_scale, 0, 255)
        aug_hsv[:, :, 2] = np.clip(aug_hsv[:, :, 2] * v_scale, 0, 255)

        # 转回BGR并保存
        aug_bgr = cv2.cvtColor(aug_hsv, cv2.COLOR_HSV2BGR)

        # 创建输出目录（如果不存在）
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

        new_img_path = os.path.join(output_dir, "images", f"{base_name}_aug{i}.jpg")
        cv2.imwrite(new_img_path, aug_bgr)

        # 复制标注文件
        label_path = os.path.join(label_dir, f"{base_name}.txt")
        new_label_path = os.path.join(output_dir, "labels", f"{base_name}_aug{i}.txt")

        if os.path.exists(label_path):
            shutil.copy(label_path, new_label_path)
        else:
            print(f"警告: 找不到标签文件 {label_path}")


# 配置路径
original_image_dir = "D:/ultralytics-8.3.33(yolo)/data/images"  # 替换为您的原始图像目录
original_label_dir = "D:/ultralytics-8.3.33(yolo)/data/labels"  # 替换为您的原始标签目录
output_dir = "D:/ultralytics-8.3.33(yolo)/data/122"  # 增强数据集输出目录

# 获取所有原始图像路径
original_images = glob.glob(os.path.join(original_image_dir, "*.jpg")) + \
                  glob.glob(os.path.join(original_image_dir, "*.png")) + \
                  glob.glob(os.path.join(original_image_dir, "*.jpeg"))

print(f"找到 {len(original_images)} 张原始图像")

# 对数据集中的每张图像生成2个增强版本
for img_path in original_images:
    hsv_augmentation(img_path, output_dir, original_label_dir, aug_count=2)

print(f"数据增强完成！增强后的图像保存在: {output_dir}")