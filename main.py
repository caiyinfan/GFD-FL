import cv2
import time
import numpy as np

#开始时间#
# start=time.time()
start = time.perf_counter()
# 读取三张连续的照片
image1 = cv2.imread('ECSP2963.JPG')
image2 = cv2.imread('ECSP2964.JPG')
image3 = cv2.imread('ECSP2965.JPG')

# 将照片转换为灰度图像
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)

# 计算光流gray1和gray2：输入的两张灰度图像。
# None：用于计算光流的窗口大小，这里设置为None表示使用默认值。
# 0.5：金字塔的缩放比例。
# 3：金字塔层数。
# 15：迭代次数。
# 3：多尺度估计的层数。
# 5：角点检测的最小响应。
# 1.2：金字塔的缩放因子。
# 0：多尺度估计的层数。
flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 35, 3, 5, 1.2, 0)

# 计算光流的幅度和角度
magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

# 设定阈值来确定运动物体
threshold = 1
motion_mask = magnitude > threshold

# 标记运动物体
image2_with_motion = image2.copy()
image2_with_motion[motion_mask] = [0, 0, 255]  # 将运动物体标记为红色

# 保存结果图像
# cv2.imwrite('24818.jpg', image2_with_motion)
# 显示结果
# cv2.imshow('Motion Detection', image2_with_motion)
cv2.waitKey(0)
cv2.destroyAllWindows()

# end=time.time()
end = time.perf_counter()
print('Running time: %s Seconds'%(end-start))