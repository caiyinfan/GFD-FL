import cv2
import time
import numpy as np
import os
from tqdm import tqdm

# 开始时间
start = time.perf_counter()


root = r'D:\RPCA\cai'
save_root = r'D:\RPCA\cai_tf'
save_root_mask = r'D:\RPCA\cai_tf_mask'

img_list = os.listdir(root)

for i in tqdm(range(0, len(img_list) - 2, 3)):
    img_names = img_list[i: i + 3]
    # 读取三张连续的照片
    image1 = cv2.imread(os.path.join(root, img_names[0]))
    image2 = cv2.imread(os.path.join(root, img_names[1]))
    image3 = cv2.imread(os.path.join(root, img_names[2]))

    # 将照片转换为灰度图像
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)

    # 计算光流
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 45, 3, 5, 1.2, 0)

    # 计算光流的幅度和角度
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # 设定阈值来确定运动物体
    threshold = 1
    motion_mask = magnitude > threshold

    # 创建一个透明度掩码
    alpha = 0.5  # 透明度，0.5表示半透明
    color_mask = np.zeros_like(image2)
    color_mask[motion_mask] = [255, 0, 0]  # 使用运动掩码的红色通道

    # 将透明度掩码应用到第二张图片上
    overlay = image2.copy()
    cv2.addWeighted(color_mask, alpha, overlay, 1 - alpha, 0, overlay)

    # 保存结果图像
    cv2.imwrite(os.path.join(save_root, img_names[1]), overlay)

    # 保存运动区域的灰度图像
    cv2.imwrite(os.path.join(save_root_mask, img_names[1].split('.')[0] + '_mask' + '.jpg'), (motion_mask * 255).astype(np.uint8))

    # 显示结果
    # cv2.imshow('FL', overlay)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# 结束时间
end = time.perf_counter()
print('Running time: %s Seconds' % (end - start))
