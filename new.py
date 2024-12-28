import cv2
import os
import re
import logging

def process_three_frame_difference(input_folder, output_folder, threshold=25):
    """
    Processes images in the specified input folder using the three-frame difference method,
    and saves the motion detected images to the output folder.

    Parameters:
    - input_folder: Path to the folder containing input images.
    - output_folder: Path to the folder where output images will be saved.
    - threshold: Threshold for motion detection in the difference images.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if not os.path.exists(input_folder):
        logging.error(f"Input folder {input_folder} does not exist.")
        return

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
                if img1 is None:
                    logging.warning(f"Could not read image: {img1_path}")
                if img2 is None:
                    logging.warning(f"Could not read image: {img2_path}")
                if img3 is None:
                    logging.warning(f"Could not read image: {img3_path}")

                # 跳过如果有任何图像未能读取
                if img1 is None or img2 is None or img3 is None:
                    logging.warning(f"Skipping group {prefix} due to unreadable images.")
                    continue

                # 检查并调整图像尺寸
                if img1.shape != img2.shape or img1.shape != img3.shape:
                    # 获取第一张图的尺寸
                    height, width = img1.shape[:2]
                    img2 = cv2.resize(img2, (width, height))
                    img3 = cv2.resize(img3, (width, height))

                # 转换为灰度图像
                gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

                # 计算差分图像
                diff1 = cv2.absdiff(gray1, gray2)
                diff2 = cv2.absdiff(gray2, gray3)

                # 应用阈值
                _, diff1_t = cv2.threshold(diff1, threshold, 255, cv2.THRESH_BINARY)
                _, diff2_t = cv2.threshold(diff2, threshold, 255, cv2.THRESH_BINARY)

                # 合并差分结果
                combined_motion = cv2.bitwise_or(diff1_t, diff2_t)

                # 将运动区域叠加在中间帧上
                result_img = cv2.addWeighted(img2, 1, combined_motion, 0.5, 0)

                # 保存结果图像
                output_file_name = f"motion_{prefix}_{group[2]}"
                cv2.imwrite(os.path.join(output_folder, output_file_name), result_img)
                logging.info(f"Saved motion detected image: {output_file_name}")
            else:
                logging.warning(f"Missing frames for group {prefix}. Skipping.")

    except Exception as e:
        logging.error(f"Error occurred: {e}")

if __name__ == "__main__":
    input_folder = r"D:\RPCA\cai"  # 修改为包含图片文件的文件夹路径
    output_folder = r"D:\RPCA\cai3"  # 替换为输出文件夹的路径
    process_three_frame_difference(input_folder, output_folder)
