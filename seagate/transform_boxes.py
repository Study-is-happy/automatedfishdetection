import numpy as np
import json
import cv2
import matplotlib.pyplot as plt

import config
import utils

stereo_params = np.load(config.calib_dir + 'stereo_params.npz')

# left_homography_matrix = np.array([[0.01754, -0.00124, 0.25097],
#                                    [0.00061, 0.01826, -0.75414],
#                                    [-0., 0., 0.01889]])

left_homography_matrix = np.eye(3)


disparity_map = np.load(config.calib_dir + 'disparity_map.npy')

disparity_map = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX)

disparity_map = cv2.imread(config.calib_dir + 'port.png')
disparity_map = cv2.cvtColor(disparity_map, cv2.COLOR_BGR2RGB)

# disparity_map = cv2.cvtColor(disparity_map, cv2.COLOR_GRAY2BGR)

with open(config.project_dir + 'train/instances.json') as instances_file:

    instances_dict = json.load(instances_file)
    instance = instances_dict['20161027.175930.00463_rect_color']
    for annotation in instance['annotations']:
        bbox = annotation['bbox']
        utils.rel_to_abs(bbox, instance['width'], instance['height'])
        points = np.array([[[bbox[0], bbox[1]],
                            [bbox[2], bbox[1]],
                            [bbox[0], bbox[3]],
                            [bbox[2], bbox[3]]]])
        distort_points = cv2.undistortPoints(points, stereo_params['left_projection'][:, :-1], np.zeros(5), R=np.linalg.inv(stereo_params['left_rotation']), P=stereo_params['left_camera_matrix'])

        distort_points = np.hstack((distort_points[0], np.array([np.ones(4)]).T))

        distort_points = left_homography_matrix.dot(distort_points.T)
        distort_points /= distort_points[-1]
        distort_points = distort_points[:-1].T

        distort_points = utils.get_rint(distort_points)
        cv2.line(disparity_map, tuple(distort_points[0]), tuple(distort_points[1]), (0, 255, 0), 2)
        cv2.line(disparity_map, tuple(distort_points[1]), tuple(distort_points[3]), (0, 255, 0), 2)
        cv2.line(disparity_map, tuple(distort_points[3]), tuple(distort_points[2]), (0, 255, 0), 2)
        cv2.line(disparity_map, tuple(distort_points[2]), tuple(distort_points[0]), (0, 255, 0), 2)

plt.imshow(disparity_map)
plt.show()
