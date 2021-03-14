import numpy as np
import json
import cv2
import matplotlib.pyplot as plt

import util

with open('/home/auv/venv3/left_projection.npy', 'rb') as left_projection_file:
    left_projection = np.load(left_projection_file)

with open('/home/auv/venv3/left_rotation.npy', 'rb') as left_rotation_file:
    left_rotation = np.load(left_rotation_file)

with open('/home/auv/venv3/left_camera_matrix.npy', 'rb') as left_camera_matrix_file:
    left_camera_matrix = np.load(left_camera_matrix_file)

homography_matrix = np.array([[0.98182, - 0.03755, - 81.06596],
                              [0.04513, 0.99161, - 54.79795],
                              [-0., - 0., 1.]])

right_image = cv2.imread('/home/auv/venv3/stbd_2.png')


with open('/home/auv/venv3/instances.json') as instances_file:

    instances_dict = json.load(instances_file)
    instance = instances_dict['20161027.175018.00256_rect_color']

    for annotation in instance['annotations']:
        bbox = annotation['bbox']
        util.rel_to_abs(bbox, instance['width'], instance['height'])
        points = np.array([[[bbox[0], bbox[1]],
                            [bbox[2], bbox[1]],
                            [bbox[0], bbox[3]],
                            [bbox[2], bbox[3]]]])

        points = cv2.undistortPoints(points, left_projection[:, :-1], np.zeros(5), R=np.linalg.inv(left_rotation), P=left_camera_matrix)
        points = np.squeeze(points)

        points = homography_matrix.dot(np.vstack((points.T, np.ones(len(points)))))

        points /= points[-1]

        points = points[:-1].T

        points = np.rint(points).astype(int)
        cv2.line(right_image, tuple(points[0]), tuple(points[1]), (0, 255, 0), 2)
        cv2.line(right_image, tuple(points[1]), tuple(points[3]), (0, 255, 0), 2)
        cv2.line(right_image, tuple(points[3]), tuple(points[2]), (0, 255, 0), 2)
        cv2.line(right_image, tuple(points[2]), tuple(points[0]), (0, 255, 0), 2)

plt.imshow(right_image)
plt.show()
