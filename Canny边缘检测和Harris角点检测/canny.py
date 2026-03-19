import cv2
import numpy as np
import os


def manual_canny(img, low_threshold, high_threshold, sigma=1):
    # 1. 高斯滤波降噪
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), sigma)

    # 2. 计算梯度 (Sobel)
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
            # 简化版方向判断：0, 45, 90, 135度
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q, r = magnitude[i, j + 1], magnitude[i, j - 1]
            elif (22.5 <= angle[i, j] < 67.5):
                q, r = magnitude[i + 1, j - 1], magnitude[i - 1, j + 1]
            elif (67.5 <= angle[i, j] < 112.5):
                q, r = magnitude[i + 1, j], magnitude[i - 1, j]
            elif (112.5 <= angle[i, j] < 157.5):
                q, r = magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]
            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                nms[i, j] = magnitude[i, j]
            else:
                nms[i, j] = 0

    # 4. 双阈值与边缘跟踪 (Hysteresis)
    res = np.zeros((M, N), dtype=np.uint8)
    strong_i, strong_j = np.where(nms >= high_threshold)
    weak_i, weak_j = np.where((nms <= high_threshold) & (nms >= low_threshold))
    res[strong_i, strong_j] = 255
    res[weak_i, weak_j] = 50

    # 简单的连接性检查
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if res[i, j] == 50:
                if (res[i - 1:i + 2, j - 1:j + 2] == 255).any():
                    res[i, j] = 255
                else:
                    res[i, j] = 0
    return res


def save_canny_tuning_manual(image_path, output_dir="canny_results_manual"):
    img = cv2.imread(image_path)
    if img is None: return
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    sigmas = [1, 2]
    threshold_pairs = [(30, 90), (70, 150), (100, 200)]

    for sigma in sigmas:
        for low, high in threshold_pairs:
            # 调用手动实现的函数，不再直接调用 cv2.Canny
            result = manual_canny(img, low, high, sigma)

            filename = f"manual_canny_s{sigma}_l{low}_h{high}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), result)
            print(f"已保存手动 Canny 结果: {filename}")


if __name__ == "__main__":
    save_canny_tuning_manual('dog.jpg')