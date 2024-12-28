import os
import cv2
from PIL import Image
from ultralytics import YOLO
import numpy as np

# 假设YOLO模型文件路径已经正确设置
model_file = "runs/detect/train/weights/best.pt"  # 请替换为实际的模型文件路径
model = YOLO(model_file)

# 指定输入图像的目录和输出目录
image_folder = r"C:\Users\cai_y\Desktop\测试"
output_folder = r"C:\Users\cai_y\Desktop\测试\测试"

# 确保输出目录存在
os.makedirs(output_folder, exist_ok=True)

# 光流法处理连续的三张图像并进行预测
def optical_flow_and_predict(images, filenames):
    try:
        # 光流法处理
        flows = []
        for i in range(len(images) - 1):
            gray1 = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(images[i + 1], cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 30, 3, 5, 1.2, 0)
            flows.append(flow)

        # 将光流结果合并
        motion_mask = np.zeros_like(images[0][:, :, 0], dtype=bool)
        for flow in flows:
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_mask |= magnitude > 1

        # 可视化运动区域到图像
        alpha = 0.5
        color_mask = np.zeros_like(images[0])
        color_mask[:, :, 2] = 255 * motion_mask * alpha  # 设置蓝色色通道为运动区域
        image_with_motion = cv2.addWeighted(images[1], 1 - alpha, color_mask, alpha, 0)

        # 将处理后的图像传递给模型进行预测
        image_with_motion_pil = Image.fromarray(image_with_motion)
        results = model(image_with_motion_pil)

        # 检查是否有检测结果
        if results and any(len(result.boxes) > 0 for result in results):  # 如果至少检测到一个目标
            # 复制图像到输出目录
            for filename, image in zip(filenames, images):
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, image)  # 保存原始图像的副本

    except Exception as e:
        print(f"处理图像时出错：{str(e)}")

# 遍历输入目录中的所有图像文件
files = os.listdir(image_folder)
sorted_files = sorted([f for f in files if f.endswith('.jpg') or f.endswith('.JPG')])

# 处理连续的三张图像并进行预测
for i in range(0, len(sorted_files) - 2, 3):  # 每次处理三张图片
    current_image = cv2.imread(os.path.join(image_folder, sorted_files[i]))
    next_image = cv2.imread(os.path.join(image_folder, sorted_files[i + 1]))
    after_next_image = cv2.imread(os.path.join(image_folder, sorted_files[i + 2]))

    images = [current_image, next_image, after_next_image]
    filenames = [sorted_files[i], sorted_files[i + 1], sorted_files[i + 2]]  # 保存文件名列表

    # 调用函数进行光流法处理和预测
    optical_flow_and_predict(images, filenames)