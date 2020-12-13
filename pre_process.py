import cv2
import numpy as np
import matplotlib.pyplot as plt

import config

image = cv2.imread(config.project_dir +
                   'train/images/20191009.161024.00745_rect_color.jpg', -1)


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype(np.uint8)
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


plt.imshow(image)
plt.show()
