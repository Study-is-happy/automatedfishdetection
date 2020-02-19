import numpy as np
import cv2
from matplotlib import pyplot as plt
# import matplotlib.image as mpimg

# # img = cv2.imread('CCXC.20051018.14494007.0036.jpg')[1:163, 942:1106]
# img = cv2.imread('CCXC.20051018.14494007.0036.jpg')[0:112, 991:1055]
# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# plt.imshow(img)
# mpimg.imsave("out.jpg", img)
# plt.show()

# # mask = np.zeros(img.shape[:2], np.uint8)
# # bgdModel = np.zeros((1, 65), np.float64)
# # fgdModel = np.zeros((1, 65), np.float64)
# # rect = (30, 0, 104, 152)

# # cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
# # mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
# # img = img*mask2[:, :, np.newaxis]

# # plt.imshow(img)
# # plt.show()


img = cv2.imread('20100920.215954.00142.jpg')[546-50:766+50, 1603-50:1716+50]

edges = cv2.Canny(img, 80, 180)  # 参数:图片，minval，maxval,kernel = 3

plt.subplot(121)  # 121表示行数，列数，图片的序号即共一行两列，第一张图
plt.imshow(img, cmap='gray')  # cmap :colormap 设置颜色
plt.title('original image'), plt.xticks([]), plt.yticks([])  # 坐标轴起点，终点的值
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('edge image'), plt.xticks([]), plt.yticks([])

plt.show()
# mask = np.zeros(img.shape[:2], np.uint8)

# bgdModel = np.zeros((1, 65), np.float64)
# fgdModel = np.zeros((1, 65), np.float64)

# rect = (50, 50, 1716-1603, 766-546)
# cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
# img = img*mask2[:, :, np.newaxis]

# plt.imshow(img)
# plt.show()
