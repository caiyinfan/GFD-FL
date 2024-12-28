import cv2
import os
import re
import logging

def process_three_frame_difference(input_folder, output_folder, threshold=30):
    """
    Processes images in the specified input folder using the three-frame difference method,
    and saves the motion detected images and masks to the output folder.

    Parameters:
    - input_folder: Path to the folder containing input images.
    - output_folder: Path to the folder where output images will be saved.
    - threshold: Threshold for motion detection in the difference images.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # 获取输入文件夹中所有图像文件并排序
        files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        logging.info(f"Files found: {files}")

        # 正则表达式提取文件名前缀和后缀（_1, _2, _3）
        pattern = re.compile(r'(.*)_(\d)\.jpg')
        grouped_files = {}

        # 将相同前缀的文件进行分组
        for file in files:
            match = pattern.match(file)
            if match:
                prefix = match.group(1)
                suffix = int(match.group(2))
                if prefix not in grouped_files:
                    grouped_files[prefix] = {}
                grouped_files[prefix][suffix] = file

        # 确保输出文件夹存在
        os.makedirs(output_folder, exist_ok=True)

        # 对每组文件（_1, _2, _3）进行处理
        for prefix, group in grouped_files.items():
            if all(k in group for k in [1, 2, 3]):  # 检查是否存在所有帧
                img1_path = os.path.join(input_folder, group[1])
                img2_path = os.path.join(input_folder, group[2])
                img3_path = os.path.join(input_folder, group[3])

                logging.info(f"Processing files: {img1_path}, {img2_path}, {img3_path}")

                # 读取图像
                img1 = cv2.imread(img1_path)
                img2 = cv2.imread(img2_path)
                img3 = cv2.imread(img3_path)

                # 检查图像是否成功读取
                if img1 is None or img2 is None or img3 is None:
                    logging.warning(f"Could not read one or more images for group {prefix}. Skipping.")
                    continue

                # 将照片转换为灰度图像
                gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

                # 计算两个连续帧之间的差异
                diff1 = cv2.absdiff(gray1, gray2)
                diff2 = cv2.absdiff(gray2, gray3)

                # 结合两个帧之间的差异
                motion_mask = cv2.bitwise_or(diff1, diff2)

                # 设定阈值来确定运动物体
                _, motion_mask = cv2.threshold(motion_mask, threshold, 255, cv2.THRESH_BINARY)

                # 保存运动掩码图像
                mask_output_path = os.path.join(output_folder, f'motion_mask_{prefix}.jpg')
                cv2.imwrite(mask_output_path, motion_mask)
                logging.info(f"Saved motion mask: {mask_output_path}")

                # 可选：将运动区域以红色高亮显示在原始图像上
                color_mask = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
                color_mask[:, :, 1] = 0  # 将绿色通道置为0
                color_mask[:, :, 2] = 0  # 将蓝色通道置为0
                result_image = cv2.addWeighted(img2, 1, color_mask, 0.5, 0)

                # 保存带有运动高亮的图像
                result_output_path = os.path.join(output_folder, f'highlighted_motion_{prefix}.jpg')
                cv2.imwrite(result_output_path, result_image)
                logging.info(f"Saved highlighted motion image: {result_output_path}")

            else:
                logging.warning(f"Missing frames for group {prefix}. Skipping.")

    except Exception as e:
        logging.error(f"Error occurred: {e}")

if __name__ == "__main__":
    input_folder = r"D:\RPCA\cai"  # 修改为包含图片文件的文件夹路径
    output_folder = r"D:\RPCA\cai3"  # 替换为输出文件夹的路径
    process_three_frame_difference(input_folder, output_folder)
