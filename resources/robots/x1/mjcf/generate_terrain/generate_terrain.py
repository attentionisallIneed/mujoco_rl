import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# 设置图片尺寸
width, height = 1080, 1080  # 你可以根据需要调整尺寸

# 生成随机地形数据
terrain = np.random.rand(width, height) * 255  # 随机灰度值

# 使用高斯滤波平滑地形，控制起伏程度
terrain = gaussian_filter(terrain, sigma=50)  # sigma越大，地形越平滑

# 正则化灰度值到0-255
terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min()) * 255

# 将地形数据保存为灰度图片
plt.imsave(r'../environment/heightfield.png', terrain, cmap='gray')
print("Heightfield image saved as 'heightfield.png'")
