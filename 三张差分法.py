import cv2
import numpy as np
import os
import logging

def process_three_frame_difference(input_folder, output_folder, threshold=25):
    """
    Processes images in the specified input folder using three-frame difference method,
    and saves the motion detected images to the output folder.

    Parameters:
    - input_folder: Path to the folder containing input images.
    - output_folder: Path to the folder where output images will be saved.
    - threshold: Threshold for motion detection in the difference images.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # 获取输入文件夹中所有图像文件并排序
        files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        sorted_files = sorted(files)

        # 确保输出文件夹存在
        os.makedirs(output_folder, exist_ok=True)

        # 按组读取三张图像
        for i in range(0, len(sorted_files) - 2, 3):
            img1_path = os.path.join(input_folder, sorted_files[i])
            img2_path = os.path.join(input_folder, sorted_files[i + 1])
            img3_path = os.path.join(input_folder, sorted_files[i + 2])

            logging.info(f"Processing files: {img1_path}, {img2_path}, {img3_path}")

            # 读取图像
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            img3 = cv2.imread(img3_path)

            if img1 is None or img2 is None or img3 is None:
                logging.warning("Some images could not be read and will be skipped.")
                continue

            # 转换为灰度图像
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

            # 计算差分图像
            diff1 = cv2.absdiff(gray1, gray2)
            diff2 = cv2.absdiff(gray2, gray3)

            # 应用阈值
            _, diff1_t = cv2.threshold(diff1, threshold, 255, cv2.THRESH_BINARY)
            _, diff2_t = cv2.threshold(diff2, threshold, 255, cv2.THRESH_BINARY)+-

            # 合并差分结果
            combined_motion = cv2.bitwise_or(diff1_t, diff2_t)

            # 将运动区域叠加在中间帧上
            result_img = cv2.addWeighted(img2, 1, combined_motion, 0.5, 0)

            # 保存结果图像
            cv2.imwrite(os.path.join(output_folder, f"motion_{sorted_files[i+1]}"), result_img)
            logging.info(f"Saved motion detected image for {sorted_files[i+1]}")

    except Exception as e:
        logging.error(f"Error occurred: {e}")

if __name__ == "__main__":
    input_folder = r"D:\光流照片"  # 修改为包含图片文件的文件夹路径
    output_folder = r"D:\光帧间差法"  # 替换为输出文件夹的路径

    process_three_frame_difference(input_folder, output_folder)