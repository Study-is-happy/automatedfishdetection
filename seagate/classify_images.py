import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
import sklearn.cluster
import shutil
import os

import config

image_mask = cv2.imread(config.calib_dir + 'port_mask.png', -1)
resize_image_mask = cv2.resize(image_mask, (600, 500))

image_dir = '/media/auv/Seagate Desktop Drive/AUV_images_fcts/SH-18-12/d20181012_1/port/port_rectified/'

image_path_list = np.array(sorted(glob.glob(image_dir + '*.tif'))[1:200])

image_data_list = []

for image_path in image_path_list:
    print(image_path)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image[image_mask == 0] = 0

    image = cv2.resize(image, (600, 500))

    image_data = image[resize_image_mask != 0].astype(np.float32)

    min_image_data = np.min(image_data)
    image_data = (image_data - min_image_data) / (np.max(image_data) - min_image_data)

    image_data_list.append(image_data)

num_clusters = 2


cluster = sklearn.cluster.KMeans(n_clusters=num_clusters)

cluster.fit(image_data_list)

for label in range(num_clusters):

    cluster_dir = config.calib_dir + config.calib_sub_dir + str(label) + '/'

    if os.path.exists(cluster_dir):
        shutil.rmtree(cluster_dir)
    os.mkdir(cluster_dir)

    for image_path in image_path_list[cluster.labels_ == label]:

        shutil.copy(image_path, cluster_dir)
