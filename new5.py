import os
from PIL import Image
from ultralytics import YOLO
import cv2

# 假设您已经有了一个训练好的 YOLO 模型文件路径
model_file = "MODELUSER/34/best.pt"  # 请替换为您的模型文件路径

# 设置模型为推理模式
model = YOLO(model_file)


def list_images_and_save(input_folder, output_folder, model):
    """
    Walks through all subfolders in the specified input folder, finds image files,
    performs object detection using YOLO, and saves the detected images based on confidence.

    Parameters:
    - input_folder: Path to the root folder to search for images.
    - output_folder: Path to the folder where detected images will be saved.
    - model: YOLO model instance for object detection.
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历input_folder及其所有子文件夹
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                # 构建输入和输出文件的完整路径
                input_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_folder)
                output_subfolder = os.path.join(output_folder, relative_path)
                output_file_path = os.path.join(output_subfolder, file)

                # 确保输出子文件夹存在
                os.makedirs(output_subfolder, exist_ok=True)

                # 读取图片并进行对象检测
                image = Image.open(input_file_path)
                results = model(image)

                # 处理预测结果，并将置信度大于0.5的目标放入指定路径中
                if results:
                    for result in results:
                        if result.boxes is not None:
                            for box in result.boxes:
                                confidence = box.conf
                                if confidence > 0.5:
                                    # 截取置信度为四位小数的字符串
                                    confidence_str = f"{confidence:.4f}"
                                    output_path = os.path.join(output_folder, confidence_str + "_" + file)

                                    # 将目标图像保存至指定路径
                                    image.save(output_path)
                                    print(f"Detected image saved: {output_path}")


if __name__ == "__main__":
    input_folder = "path_to_your_root_folder"  # 修改为包含图片文件的根文件夹路径
    output_folder = "path_to_your_output_folder"  # 修改为输出文件夹的路径

    list_images_and_save(input_folder, output_folder, model)
