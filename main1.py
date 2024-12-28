import cv2
import time

# 开始计时
# start = time.time()
start = time.perf_counter()
# 读取两张连续的照片
image1 = cv2.imread('ECSP2963.JPG')
image2 = cv2.imread('ECSP2964.JPG')

# 将照片转换为灰度图像
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# 计算第一帧和第二帧之间的差值
frameDelta = cv2.absdiff(gray1, gray2)

# 二值化处理
thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

# 形态学操作去除噪点
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)

# 查找轮廓
contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 在第二张图片上标记运动物体
for contour in contours:
    if cv2.contourArea(contour) < 225:
        continue
    (x, y, w, h) = cv2.boundingRect(contour)
    cv2.rectangle(image2, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 显示结果
# cv2.imshow('Motion Detection', image2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 结束计时并打印运行时间
# end = time.time()
end = time.perf_counter()
print('Running time: %s Seconds' % (end - start))