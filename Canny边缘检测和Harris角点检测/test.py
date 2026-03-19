import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# ==========================================
# 算法实现部分
# ==========================================

def harris_corner_detection(image, k=0.04, window_size=3, threshold_ratio=0.01):
    """
    Harris 角点检测的自定义实现。
    """
    if image is None: return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # 1. 计算图像梯度
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # 2. 构造结构张量的元素
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix * Iy

    # 3. 高斯平滑窗口
    # 注意：window_size 必须是奇数
    blur_ksize = window_size
    Sxx = cv2.GaussianBlur(Ixx, (blur_ksize, blur_ksize), 0)
    Syy = cv2.GaussianBlur(Iyy, (blur_ksize, blur_ksize), 0)
    Sxy = cv2.GaussianBlur(Ixy, (blur_ksize, blur_ksize), 0)

    # 4. 计算 Harris 响应 R
    det_M = (Sxx * Syy) - (Sxy ** 2)
    trace_M = Sxx + Syy
    R = det_M - k * (trace_M ** 2)

    # 5. 阈值处理与非极大值抑制 (NMS)
    threshold = threshold_ratio * R.max()
    corner_image = np.copy(image)

    # 寻找局部最大值 (简单的 NMS)
    R_max = cv2.dilate(R, None)
    corner_mask = (R == R_max) & (R > threshold)

    # 将角点标记为红色
    corner_image[corner_mask] = [0, 0, 255]
    return corner_image


def canny_edge_detection(img, low_threshold, high_threshold, sigma=1):
    """
    Canny 边缘检测的自定义实现 (简化版流程)。
    """
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. 高斯滤波降噪
    blurred = cv2.GaussianBlur(gray, (5, 5), sigma)

    # 2. 计算梯度幅度和方向
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    angle = np.arctan2(grad_y, grad_x) * 180 / np.pi
    angle[angle < 0] += 180

    # 3. 非极大值抑制 (NMS)
    M, N = gray.shape
    nms = np.zeros((M, N), dtype=np.float32)
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            q, r = 255, 255
            # 根据梯度方向找相邻像素进行比较
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):  # 0度 (水平)
                q, r = magnitude[i, j + 1], magnitude[i, j - 1]
            elif (22.5 <= angle[i, j] < 67.5):  # 45度
                q, r = magnitude[i + 1, j - 1], magnitude[i - 1, j + 1]
            elif (67.5 <= angle[i, j] < 112.5):  # 90度 (垂直)
                q, r = magnitude[i + 1, j], magnitude[i - 1, j]
            elif (112.5 <= angle[i, j] < 157.5):  # 135度
                q, r = magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]

            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                nms[i, j] = magnitude[i, j]
            else:
                nms[i, j] = 0

    # 4. 双阈值检测 (确定强弱边缘)
    res = np.zeros((M, N), dtype=np.int32)
    strong_i, strong_j = np.where(nms >= high_threshold)
    weak_i, weak_j = np.where((nms <= high_threshold) & (nms >= low_threshold))
    res[strong_i, strong_j] = 255
    res[weak_i, weak_j] = 50  # 弱边缘临时标记

    # 5. 滞后边界跟踪 (Hysteresis)
    # (简化版：检查弱边缘周围8邻域是否有强边缘)
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if res[i, j] == 50:
                if (res[i - 1:i + 2, j - 1:j + 2] == 255).any():
                    res[i, j] = 255  # 如果有强边缘，保留
                else:
                    res[i, j] = 0  # 如果没有强边缘，抑制
    return res.astype(np.uint8)


# ==========================================
# 实验主程序：参数对比实验
# ==========================================

def run_experiments():
    # --------------------------------------
    # 实验一：Harris 角点检测对比图
    # --------------------------------------
    img_temple = cv2.imread('temple.jpg')
    if img_temple is not None:
        print("正在进行 Harris 实验...")

        # 定义要测试的参数组合 [(Sigma/WindowSize, k_value)]
        harris_params = [
            (3, 0.04),  # 默认参数
            (3, 0.15),  # 增大 k (降低敏感度)
            (9, 0.04),  # 增大窗口大小 (平滑度更高)
            (9, 0.15)  # 同时增大窗口和 k
        ]

        plt.figure(figsize=(15, 10))
        plt.suptitle("Experiment 1: Harris Parameter Comparison", fontsize=16)

        for i, (win, kv) in enumerate(harris_params):
            result = harris_corner_detection(img_temple, k=kv, window_size=win)

            # 绘图配置
            plt.subplot(2, 2, i + 1)
            plt.title(f"Window Size: {win}, k: {kv}")
            # 注意：matplotlib 接收 RGB，Opencv 读出 BGR
            plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            plt.axis('off')  # 隐藏坐标轴

        # --------------------------------------
        # 实验二：Canny 边缘检测对比图
        # --------------------------------------
        img_dog = cv2.imread('dog.jpg')
        if img_dog is not None:
            print("正在进行 Canny 实验...")

            # 定义要测试的参数组合 [(Sigma, LowT, HighT)]
            canny_params = [
                (1, 50, 150),  # 默认参数
                (1, 100, 200),  # 提高整体阈值 (抑制背景噪声)
                (3, 50, 150),  # 增大 Sigma (模糊度提高)
                (3, 100, 200)  # 同时增大 Sigma 和阈值
            ]

            plt.figure(figsize=(15, 10))
            plt.suptitle("Experiment 2: Canny Parameter Comparison", fontsize=16)

            for i, (sig, lt, ht) in enumerate(canny_params):
                result = canny_edge_detection(img_dog, low_threshold=lt, high_threshold=ht, sigma=sig)

                # 绘图配置
                plt.subplot(2, 2, i + 1)
                plt.title(f"Gaussian Sigma: {sig}, T: [{lt}, {ht}]")
                plt.imshow(result, cmap='gray')  # Canny 输出是灰度图
                plt.axis('off')


        # 导入你之前实现的 canny_edge_detection 函数
        # 这里假设函数名与之前一致



        # 显示所有图表
        print("实验完成，正在绘制...")
        plt.show()
    else:
        print("错误：无法找到 temple.jpg 或 dog.jpg 图片，请确保图片已保存至当前目录。")








if __name__ == "__main__":
    run_experiments()