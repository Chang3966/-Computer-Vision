import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
import cv2

# 1. 加载图像（以 skimage 自带的人脸图像为例，也可以用 cv2.imread 加载自己的图）
image = data.astronaut()
image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# 2. 计算 HOG 特征
# orientations: 直方图的槽数（Bin）0-180度，每20度为一个bin
# pixels_per_cell: 细胞单元的大小
# cells_per_block: 块的大小
# visualize: 是否返回用于显示的 HOG 图像
fd, hog_image = hog(image_gray,
                    orientations=9,
                    pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2),
                    visualize=True,
                    channel_axis=None)

# 3. 结果可视化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image_gray, cmap=plt.cm.gray)
ax1.set_title('Original Image')

# 对 HOG 图像进行强度缩放，方便观察
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('HOG Visualization')

plt.show()

print(f"特征向量的维度是: {fd.shape}")