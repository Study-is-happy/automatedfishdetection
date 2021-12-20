import numpy as np
import cv2

sift = cv2.SIFT_create()

left_image = cv2.imread('/home/auv/Figure_1.png')
right_image = cv2.imread('/home/auv/Figure_2.png')

sift.detectAndCompute(left_image, None)
