import cv2
import numpy as np
import os


def harris_core(img, k, win, threshold_ratio):
    """核心算法逻辑：返回标记好角点的图片"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    Ixx = cv2.GaussianBlur(Ix ** 2, (win, win), 0)
    Iyy = cv2.GaussianBlur(Iy ** 2, (win, win), 0)
    Ixy = cv2.GaussianBlur(Ix * Iy, (win, win), 0)

    det_M = (Ixx * Iyy) - (Ixy ** 2)
    trace_M = Ixx + Iyy
    R = det_M - k * (trace_M ** 2)

    R_max = cv2.dilate(R, None)
    threshold = threshold_ratio * R.max()
    corner_mask = (R == R_max) & (R > threshold)

    result = np.copy(img)
    # 使用坐标画圆，看得更清楚
    coords = np.argwhere(corner_mask)
    for y, x in coords:
        cv2.circle(result, (x, y), 2, (0, 0, 255), -1)
    return result, len(coords)


def run_parameter_experiments(image_path):
    img = cv2.imread(image_path)
    if img is None: return print("找不到图片")

    output_dir = "harris_experiments"
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # --- 实验参数设置 ---
    # 1. 测试 k 的影响 (保持 win=3, ratio=0.0005)
    k_list = [0.02, 0.04, 0.08, 0.15]

    # 2. 测试 Window Size 的影响 (保持 k=0.04, ratio=0.0005)
    win_list = [3, 5, 9, 15]

    # 3. 测试 Threshold Ratio 的影响 (保持 k=0.04, win=3)
    ratio_list = [0.0001, 0.0005, 0.005, 0.05]

    print("开始生成实验对比图...")

    # 执行实验 1: 改变 k
    for k in k_list:
        res, count = harris_core(img, k, 3, 0.0005)
        cv2.imwrite(f"{output_dir}/exp_k_{k}.png", res)
        print(f"已保存 k={k}, 角点数: {count}")

    # 执行实验 2: 改变 Window Size
    for win in win_list:
        res, count = harris_core(img, 0.04, win, 0.0005)
        cv2.imwrite(f"{output_dir}/exp_win_{win}.png", res)
        print(f"已保存 win={win}, 角点数: {count}")

    # 执行实验 3: 改变 Threshold
    for ratio in ratio_list:
        res, count = harris_core(img, 0.04, 3, ratio)
        cv2.imwrite(f"{output_dir}/exp_ratio_{ratio}.png", res)
        print(f"已保存 ratio={ratio}, 角点数: {count}")

    print(f"\n✅ 所有实验完成！请查看文件夹: {output_dir}")


if __name__ == "__main__":
    run_parameter_experiments('temple.jpg')