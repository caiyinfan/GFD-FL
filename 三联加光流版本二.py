import cv2
import numpy as np
import os
import logging

def extract_and_save_images(input_folder, output_folder, flow_output_folder, threshold=1):
    """
    Processes images in the specified input folder, calculates optical flow,
    and saves the results to the output folder.

    Parameters:
    - input_folder: Path to the folder containing input images.
    - output_folder: Path to the folder where output images will be saved.
    - flow_output_folder: Path to the folder where flow images will be saved.
    - threshold: Threshold for motion detection.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # 获取输入文件夹中所有图像文件
        files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        sorted_files = sorted(files)

        # 确保输出文件夹存在
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(flow_output_folder, exist_ok=True)

        for i in range(0, len(sorted_files) - 2, 3):
            img1_path = os.path.join(input_folder, sorted_files[i])
            img2_path = os.path.join(input_folder, sorted_files[i + 1])
            img3_path = os.path.join(input_folder, sorted_files[i + 2])

            logging.info(f"Processing files: {img1_path}, {img2_path}, {img3_path}")

            if not os.path.isfile(img1_path) or not os.path.isfile(img2_path) or not os.path.isfile(img3_path):
                logging.warning(f"File not found: {img1_path}, {img2_path}, {img3_path}")
                continue

            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            img3 = cv2.imread(img3_path)

            if img1 is None or img2 is None or img3 is None:
                logging.warning(f"Could not read one or more images: {img1_path}, {img2_path}, {img3_path}")
                continue

            # 保存提取的三张图像到指定文件夹
            cv2.imwrite(os.path.join(output_folder, sorted_files[i]), img1)
            cv2.imwrite(os.path.join(output_folder, sorted_files[i + 1]), img2)
            cv2.imwrite(os.path.join(output_folder, sorted_files[i + 2]), img3)

            # 计算光流图像
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

            flow1 = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 35, 3, 5, 1.2, 0)
            flow2 = cv2.calcOpticalFlowFarneback(gray2, gray3, None, 0.5, 3, 35, 3, 5, 1.2, 0)

            magnitude1, _ = cv2.cartToPolar(flow1[..., 0], flow1[..., 1])
            magnitude2, _ = cv2.cartToPolar(flow2[..., 0], flow2[..., 1])

            motion_mask1 = magnitude1 > threshold
            motion_mask2 = magnitude2 > threshold

            motion_mask = np.logical_and(motion_mask1, motion_mask2)

            # 透明度和标记颜色的应用
            alpha = 0.5  # 透明度，0.5表示半透明
            color_mask = np.zeros_like(img2)
            color_mask[:, :, 0] = 255 * motion_mask.astype(np.uint8)  # 使用运动掩码的蓝色通道

            # 将标记的颜色叠加到原始图像上
            masked_img = cv2.addWeighted(img2, 1 - alpha, color_mask, alpha, 0)

            # 保存光流图像和带遮罩的图像到指定文件夹，并与被遮罩的图像同名
            cv2.imwrite(os.path.join(flow_output_folder, sorted_files[i + 1]), masked_img)
            logging.info(f"Processed {sorted_files[i:i + 3]} and saved as {sorted_files[i + 1]}")

    except Exception as e:
        logging.error(f"Error occurred: {e}")

if __name__ == "__main__":
    input_folder = r"D:\光流照片"  # 修改为包含图片文件的文件夹路径
    output_folder = r"D:\光输出文件夹路径" # 修改为输出文件夹的路径
    flow_output_folder = r"D:\光帧间差法"  # 修改为保存光流图像的文件夹路径

    extract_and_save_images(input_folder, output_folder, flow_output_folder)
