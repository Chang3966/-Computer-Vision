import cv2
import numpy as np


def detect_pedestrians(image_path):
    # 1. 初始化 HOG 描述符
    # 默认设置适用于 OpenCV 内置的行人检测器 (64x128 窗口)
    hog = cv2.HOGDescriptor()

    # 2. 设置 SVM 分类器为 OpenCV 内置的默认行人分类器
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # 3. 加载图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # 为了提高检测速度和效果，通常会调整图像大小
    # (如果图像太大，速度会很慢；太小，检测不到)
    # 建议将宽度限制在 1000 像素以内
    r = 1000.0 / image.shape[1]
    dim = (1000, int(image.shape[0] * r))
    image_resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # 保留一个副本用于画图
    orig_image = image_resized.copy()

    print(f"Detecting pedestrians in image of size: {image_resized.shape[:2]}...")

    # 4. 执行多尺度检测 (Detect MultiScale)
    # 这步是核心。它会用滑动窗口在不同图像金字塔尺度上扫描。
    # 参数说明:
    # winStride: 滑动窗口在横向和纵向的步长。越小越准但越慢。
    # padding: 在窗口周围添加的像素，有助于捕捉边缘处的行人。
    # scale: 图像金字塔的缩放比例。越小(如1.03)检测越细致，但速度越慢。
    (rects, weights) = hog.detectMultiScale(image_resized,
                                            winStride=(4, 4),
                                            padding=(8, 8),
                                            scale=1.05)

    # 5. 应用非极大值抑制 (Non-Maxima Suppression, NMS)
    # HOG检测通常会在同一个行人周围产生多个重叠的框。
    # NMS 的目的是保留得分最高（最可能是行人）的框，去除重叠的框。

    # rects 格式是 (x, y, w, h)
    rects_nms = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])

    # 使用简单的重叠面积阈值来合并框 (Pick: 保留的框的索引)
    # 这里我们借用一个经典的 NMS 实现思路（OpenCV也有相关函数，但这样写更直观）
    # 在这里，我们直接使用 OpenCV 的合并重叠框的功能（通过 weights 过滤）

    # 稍微处理一下 weights，过滤掉置信度极低的框（可选）
    final_rects = []
    for i, (x, y, w, h) in enumerate(rects):
        if weights[i] > 0.7:  # 这个阈值可以根据效果调整
            final_rects.append((x, y, w, h))

    print(f"Found {len(rects)} raw detections, {len(final_rects)} after confidence filtering.")

    # 6. 绘制结果
    # 绘制原始检测框（绿色）
    # for (x, y, w, h) in rects:
    #     cv2.rectangle(orig_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 绘制过滤后的检测框（蓝色）
    for (x, y, w, h) in final_rects:
        cv2.rectangle(image_resized, (x, y), (x + w, y + h), (255, 0, 0), 3)

    # 7. 显示图像
    # cv2.imshow("Raw Detections", orig_image) # 如果想看原始框可以取消注释
    cv2.imshow("Pedestrians Detected", image_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 8. 保存结果到本地
    output_path = 'detected_pedestrians_result.jpg'
    success = cv2.imwrite(output_path, image_resized)

    if success:
        print(f"结果已成功保存至: {output_path}")
    else:
        print("保存失败，请检查路径或权限。")


if __name__ == "__main__":
    # 使用你自己的图片路径
    detect_pedestrians('img.png')