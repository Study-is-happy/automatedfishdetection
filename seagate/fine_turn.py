import config
import numpy as np
import cv2
import gtsam
import os
from scipy.spatial.transform import Rotation

unrectify_params = np.load(config.calib_dir + 'unrectify_params.npz')
left_camera_matrix = unrectify_params['left_camera_matrix']
right_camera_matrix = unrectify_params['right_camera_matrix']

global_stereo_params = np.load(config.calib_dir + config.calib_sub_dir + 'global_stereo_params.npz')
global_rotation_matrix = global_stereo_params['rotation_matrix']
global_translation = np.array([global_stereo_params['translation']]).T

inv_global_rotation_matrix = np.linalg.inv(global_rotation_matrix)
inv_global_translation = -inv_global_rotation_matrix.dot(global_translation)

symbol_X = gtsam.symbol_shorthand.X
symbol_L = gtsam.symbol_shorthand.L

gtsam_left_camera_matrix = gtsam.Cal3_S2(left_camera_matrix[0, 0], left_camera_matrix[1, 1], 0,
                                         left_camera_matrix[0, 2], left_camera_matrix[1, 2])

gtsam_right_camera_matrix = gtsam.Cal3_S2(right_camera_matrix[0, 0], right_camera_matrix[1, 1], 0,
                                          right_camera_matrix[0, 2], right_camera_matrix[1, 2])

measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.005, 0.005, 0.005, 1.0, 1.0, 1.0]))
point_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)

left_projection_matrix = left_camera_matrix.dot(np.hstack((np.eye(3), np.zeros((3, 1)))))
right_projection_matrix = right_camera_matrix.dot(np.hstack((global_rotation_matrix, global_translation)))

filtered_match_points_dir = config.calib_dir + config.calib_sub_dir + 'filtered_match_points/'

for match_points_file in sorted(os.listdir(filtered_match_points_dir)):

    match_points = np.load(filtered_match_points_dir + match_points_file)

    print(match_points_file)

    left_points = match_points['left_points']
    right_points = match_points['right_points']

    print(len(left_points))

    graph = gtsam.NonlinearFactorGraph()

    pose_factor = gtsam.PriorFactorPose3(symbol_X(0), gtsam.Pose3(), pose_noise)
    graph.push_back(pose_factor)

    object_points = cv2.triangulatePoints(left_projection_matrix, right_projection_matrix, left_points.T, right_points.T)

    object_points = (object_points / object_points[-1])[:-1].T
    # inlier_mask = object_points[:, 2] > 0
    # object_points = object_points[inlier_mask]
    # left_points = left_points[inlier_mask]
    # right_points = right_points[inlier_mask]

    point_factor = gtsam.PriorFactorPoint3(symbol_L(0), gtsam.Point3(
        object_points[0]), point_noise)
    graph.push_back(point_factor)

    initial = gtsam.Values()

    initial.insert(symbol_X(0), gtsam.Pose3())

    initial.insert(symbol_X(1), gtsam.Pose3(gtsam.Rot3(inv_global_rotation_matrix),
                                            gtsam.Point3(inv_global_translation.flatten())))

    for index, (left_point, right_point, object_point) in enumerate(zip(left_points, right_points, object_points)):

        left_factor = gtsam.GenericProjectionFactorCal3_S2(
            left_point, measurement_noise, symbol_X(0), symbol_L(index), gtsam_left_camera_matrix)
        graph.push_back(left_factor)

        right_factor = gtsam.GenericProjectionFactorCal3_S2(
            right_point, measurement_noise, symbol_X(1), symbol_L(index), gtsam_right_camera_matrix)
        graph.push_back(right_factor)

        initial.insert(symbol_L(index), gtsam.Point3(object_point))

    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)

    print(optimizer.error())
    result = optimizer.optimize()
    print(optimizer.error())

    transformation_0 = result.atPose3(symbol_X(0)).matrix()
    transformation_1 = result.atPose3(symbol_X(1)).matrix()

    transformation_matrix = transformation_0.dot(np.linalg.inv(transformation_1))
    rotation_matrix = transformation_matrix[:3, :3]
    translation = transformation_matrix[:3, 3]

    scale = cv2.norm(unrectify_params['opt_translation'][:2]) / cv2.norm(translation[:2])
    translation *= scale

    # print(Rotation.from_matrix(rotation_matrix).as_euler('zyx', degrees=True))
    print(translation)
    print()

    image_size = tuple(unrectify_params['image_size'])

    left_rotation, right_rotation, left_projection, right_projection, Q, left_roi, right_roi = cv2.stereoRectify(left_camera_matrix, np.zeros(5),
                                                                                                                 right_camera_matrix, np.zeros(5),
                                                                                                                 image_size, rotation_matrix, translation)

    left_mapx, left_mapy = cv2.initUndistortRectifyMap(left_camera_matrix, np.zeros(5), left_rotation, left_projection, image_size, cv2.CV_32F)
    right_mapx, right_mapy = cv2.initUndistortRectifyMap(right_camera_matrix, np.zeros(5), right_rotation, right_projection, image_size, cv2.CV_32F)

    np.savez(config.calib_dir + config.calib_sub_dir + 'stereo_params/' + match_points_file,
             rotation_matrix=rotation_matrix,
             translation=translation,
             left_rotation=left_rotation,
             right_rotation=right_rotation,
             left_projection=left_projection,
             right_projection=right_projection,
             Q=Q,
             left_mapx=left_mapx,
             left_mapy=left_mapy,
             right_mapx=right_mapx,
             right_mapy=right_mapy,
             left_roi=left_roi,
             right_roi=right_roi)
