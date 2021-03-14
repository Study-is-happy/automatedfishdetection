import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

import config
import utils

calib_file = cv2.FileStorage(config.calib_dir + 'calib.xml', cv2.FILE_STORAGE_READ)

image_width = int(calib_file.getNode('image_width').real())
image_height = int(calib_file.getNode('image_height').real())

image_size = (image_width, image_height)

left_camera_matrix = calib_file.getNode('left_camera_matrix').mat()
left_distortion_coefficients = calib_file.getNode('left_distortion_coefficients').mat()
right_camera_matrix = calib_file.getNode('right_camera_matrix').mat()
right_distortion_coefficients = calib_file.getNode('right_distortion_coefficients').mat()

translation = calib_file.getNode('translation').mat()
rotation_matrix = calib_file.getNode('rotation_matrix').mat()

calib_file.release()

left_rotation, right_rotation, left_projection, right_projection, Q, left_roi, right_roi = cv2.stereoRectify(left_camera_matrix, left_distortion_coefficients,
                                                                                                             right_camera_matrix, right_distortion_coefficients,
                                                                                                             image_size, rotation_matrix, translation, flags=0)


np.savez(config.calib_dir + 'stereo_params', left_camera_matrix=left_camera_matrix, left_rotation=left_rotation, left_projection=left_projection, Q=Q)


def unrectify(image, mask_image, rotation, projection, camera_matrix, roi, annotation_id=None):

    inv_mapx, inv_mapy = cv2.initUndistortRectifyMap(projection[:, :-1], np.zeros(5), np.linalg.inv(rotation), camera_matrix, image_size, cv2.CV_32F)

    mask_image[mask_image == 255] = 0
    image[mask_image == 0] = 0

    image = cv2.remap(image, inv_mapx, inv_mapy, cv2.INTER_LINEAR)

    plt.imshow(image)
    plt.show()

    if annotation_id is not None:

        with open(config.project_dir + 'train/instances.json') as instances_file:

            instances_dict = json.load(instances_file)
            instance = instances_dict[annotation_id]

            for annotation in instance['annotations']:
                bbox = annotation['bbox']
                utils.rel_to_abs(bbox, instance['width'], instance['height'])

                bbox = utils.get_rint(bbox)

                cv2.rectangle(mask_image, tuple(bbox[0:2]), tuple(bbox[2:4]), 255, -1)

    mask_image = cv2.remap(mask_image, inv_mapx, inv_mapy, cv2.INTER_NEAREST)

    roi_mask_image = np.zeros(mask_image.shape)
    roi_x, roi_y, roi_w, roi_h = roi
    roi_mask_image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = 255

    mask_image[roi_mask_image == 0] = 0

    # plt.imshow(mask_image)
    # plt.show()

    return image, mask_image


left_image, left_mask_image = unrectify(cv2.imread(config.calib_dir + '20161027.175930.00463_rect_color.jpg'),
                                        cv2.imread(config.calib_dir + 'port_mask.bmp', -1),
                                        left_rotation, left_projection, left_camera_matrix,
                                        left_roi, annotation_id='20161027.175930.00463_rect_color')

right_image, right_mask_image = unrectify(cv2.imread(config.calib_dir + '20161027.175930.00462_rect_color.jpg'),
                                          cv2.imread(config.calib_dir + 'stbd_mask.bmp', -1),
                                          right_rotation, right_projection, right_camera_matrix,
                                          right_roi)


cv2.imwrite(config.calib_dir + 'port.png', left_image)
cv2.imwrite(config.calib_dir + 'port_mask.png', left_mask_image)
cv2.imwrite(config.calib_dir + 'stbd.png', right_image)
cv2.imwrite(config.calib_dir + 'stbd_mask.png', right_mask_image)
