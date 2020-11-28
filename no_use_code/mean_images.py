import os
import cv2
import numpy as np

import config

images_dirs = [config.project_dir+'pacstorm/images/', ]

mean_bgr = np.zeros(3)
count = 0

for images_dir in images_dirs:
    for image_file_name in os.listdir(images_dir):

        image = cv2.imread(images_dir+image_file_name)

        image = cv2.resize(image, (1024, 1024))
        current_mean_bgr = np.array(cv2.mean(image)[:3])

        count += 1

        mean_bgr = mean_bgr/count*(count-1)+current_mean_bgr/count
        print('mean: '+str(mean_bgr) + ' count: '+str(count))

print(mean_bgr)
