import cv2
import numpy as np
import matplotlib.pyplot as plt

import config
import utils


def get_roi_mask(image_shape, roi):

    mask = np.zeros(image_shape, dtype=np.uint8)

    roi_x, roi_y, roi_w, roi_h = roi
    mask[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = 255

    return mask


def get_non_max_suppression_mask(keypoints, image_shape):
    binary_image = np.zeros(image_shape)
    response_list = np.array([keypoint.response for keypoint in keypoints])
    mask = np.flip(np.argsort(response_list))
    point_list = utils.get_rint([keypoint.pt for keypoint in keypoints])[mask]
    non_max_suppression_mask = []
    for point, index in zip(point_list, mask):
        if binary_image[point[1], point[0]] == 0:
            non_max_suppression_mask.append(index)
            cv2.circle(binary_image, (point[0], point[1]), 5, 255, -1)

    return non_max_suppression_mask


def get_match_points(detector, gray_left_image, gray_right_image, left_roi, right_roi):

    left_keypoints, left_descriptors = detector.detectAndCompute(gray_left_image, left_roi)
    right_keypoints, right_descriptors = detector.detectAndCompute(gray_right_image, right_roi)

    return left_keypoints, right_keypoints, left_descriptors, right_descriptors


def get_good_match_points(src_keypoints, dst_keypoints, src_descriptors, dst_descriptors):

    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = bf_matcher.knnMatch(src_descriptors, dst_descriptors, k=2)
    cross_matches = bf_matcher.match(dst_descriptors, src_descriptors)

    cross_match_dict = {}
    for cross_match in cross_matches:
        cross_match_dict[cross_match.trainIdx] = cross_match.queryIdx

    src_points = []
    dst_points = []
    distances = []

    for k_matches in matches:
        if len(k_matches) == 2:
            match_1, match_2 = k_matches
            if match_1.queryIdx in cross_match_dict and cross_match_dict[match_1.queryIdx] == match_1.trainIdx and match_1.distance < 0.75 * match_2.distance:
                # if match_1.distance < 0.8 * match_2.distance:
                src_points.append(src_keypoints[match_1.queryIdx].pt)
                dst_points.append(dst_keypoints[match_1.trainIdx].pt)
                distances.append(match_1.distance)

    return np.array(src_points), np.array(dst_points), np.array(distances)


def multi_scale_match(src_keypoints_list, dst_keypoints_list, src_descriptors_list, dst_descriptors_list, src_image, dst_image):

    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    all_src_points = []
    all_dst_points = []
    all_distances = []

    for src_keypoints, src_descriptors, dst_keypoints, dst_descriptors in zip(src_keypoints_list, src_descriptors_list, dst_keypoints_list, dst_descriptors_list):

        src_points, dst_points, distances = get_good_match_points(src_keypoints, dst_keypoints, src_descriptors, dst_descriptors)

        all_src_points.extend(src_points)
        all_dst_points.extend(dst_points)
        all_distances.extend(distances)

        # print(unpack_sift_octave(src_keypoints[match_1.queryIdx].octave))
        # plot_match_points(src_image, dst_image, src_points, dst_points, True)

    return np.array(all_src_points), np.array(all_dst_points), np.array(all_distances)


def get_non_max_suppression_match_mask(left_points, right_points, distance_list, image_shape):

    left_binary_image = np.zeros(image_shape)
    right_binary_image = np.zeros(image_shape)

    mask = np.flip(np.argsort(distance_list))
    left_points = utils.get_rint(left_points)[mask]
    right_points = utils.get_rint(right_points)[mask]
    non_max_suppression_mask = []

    for left_point, right_point, index in zip(left_points, right_points, mask):
        if left_binary_image[left_point[1], left_point[0]] == 0 and right_binary_image[right_point[1], right_point[0]] == 0:
            non_max_suppression_mask.append(index)
            cv2.circle(left_binary_image, (left_point[0], left_point[1]), 10, 255, -1)
            cv2.circle(right_binary_image, (right_point[0], right_point[1]), 10, 255, -1)

    return non_max_suppression_mask


def get_homography_match_points(detector, gray_left_image, gray_right_image, left_roi, right_roi):

    left_keypoints, right_keypoints, left_descriptors, right_descriptors = get_match_points(detector,
                                                                                            gray_left_image,
                                                                                            gray_right_image,
                                                                                            left_roi,
                                                                                            right_roi)

    bf_matcher = cv2.BFMatcher(cv2.NORM_L2)

    matches = bf_matcher.knnMatch(left_descriptors, right_descriptors, k=2)
    cross_matches = bf_matcher.match(right_descriptors, left_descriptors)

    cross_match_dict = {}
    for cross_match in cross_matches:
        cross_match_dict[cross_match.trainIdx] = cross_match.queryIdx

    left_points = []
    right_points = []
    distance_list = []

    for match_1, match_2 in matches:
        if match_1.queryIdx in cross_match_dict and cross_match_dict[match_1.queryIdx] == match_1.trainIdx and match_1.distance < 0.75 * match_2.distance:
            left_points.append(left_keypoints[match_1.queryIdx].pt)
            right_points.append(right_keypoints[match_1.trainIdx].pt)
            distance_list.append(match_1.distance)

    ransac_reproj_threshold = 2.0

    homography_matrix, homography_mask = cv2.findHomography(np.array(left_points), np.array(right_points), cv2.RANSAC, ransac_reproj_threshold)

    homography_mask = (homography_mask.ravel() == 1)
    print(np.count_nonzero(homography_mask))
    max_distance = np.max(np.array(distance_list)[homography_mask])

    left_points = []
    right_points = []
    distances = []

    for k_matches in matches:

        left_point = left_keypoints[k_matches[0].queryIdx].pt

        warp_left_point = homography_matrix.dot(np.append(left_point, [1]).T)
        warp_left_point = (warp_left_point / warp_left_point[-1])[:2]

        best_error = ransac_reproj_threshold
        best_right_point = None

        for match in k_matches:
            if match.distance <= max_distance:
                right_point = right_keypoints[match.trainIdx].pt

                error = cv2.norm(warp_left_point - right_point)
                if error < best_error:
                    best_error = error
                    best_right_point = right_point
                    best_distance = match.distance

        if best_right_point is not None:
            left_points.append(left_point)
            right_points.append(best_right_point)
            distances.append(best_distance)

    # match_mask = get_non_max_suppression_match_mask(left_points, right_points, distances, gray_left_image.shape)

    # left_points = np.array(left_points)[match_mask]
    # right_points = np.array(right_points)[match_mask]

    return np.array(left_points), np.array(right_points)


def get_fundamental_match_points(detector, gray_left_image, gray_right_image, left_roi, right_roi, ransac_reproj_threshold=3):

    left_keypoints, right_keypoints, left_descriptors, right_descriptors = get_match_points(detector,
                                                                                            gray_left_image,
                                                                                            gray_right_image,
                                                                                            left_roi,
                                                                                            right_roi)

    if len(left_keypoints) < 2 and len(right_keypoints) < 2:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_L2)

    matches = matcher.knnMatch(left_descriptors, right_descriptors, k=5)

    left_points = []
    right_points = []
    distance_list = []

    for match_1, match_2, _, _, _ in matches:
        if match_1.distance < 0.75 * match_2.distance:
            left_points.append(left_keypoints[match_1.queryIdx].pt)
            right_points.append(right_keypoints[match_1.trainIdx].pt)
            distance_list.append(match_1.distance)

    fundamental_matrix, fundamental_mask = cv2.findFundamentalMat(np.array(left_points), np.array(right_points), cv2.FM_RANSAC, ransac_reproj_threshold)

    fundamental_mask = (fundamental_mask.ravel() == 1)
    max_distance = np.max(np.array(distance_list)[fundamental_mask])

    left_points = np.array(left_points)[fundamental_mask]
    right_points = np.array(right_points)[fundamental_mask]

    # left_points = []
    # right_points = []
    # distances = []

    # for k_matches in matches:

    #     left_point = left_keypoints[k_matches[0].queryIdx].pt

    #     line = fundamental_matrix.dot(np.append(left_point, [1]).T)

    #     best_error = ransac_reproj_threshold
    #     best_right_point = None

    #     for match in k_matches:
    #         if match.distance <= max_distance:

    #             right_point = right_keypoints[match.trainIdx].pt

    #             error = np.abs(line[0] * right_point[0] + line[1] * right_point[1] + line[2]) / np.sqrt(line[0]**2 + line[1]**2)

    #             if error < best_error:
    #                 best_error = error
    #                 best_right_point = right_point
    #                 best_distance = match.distance

    #     if best_right_point is not None:
    #         left_points.append(left_point)
    #         right_points.append(best_right_point)
    #         distances.append(best_distance)

    # non_max_suppression_match_mask = get_non_max_suppression_match_mask(left_points, right_points, distances, gray_left_image.shape)

    # left_points = np.array(left_points)[non_max_suppression_match_mask]
    # right_points = np.array(right_points)[non_max_suppression_match_mask]

    return left_points, right_points


def plot_match_points(left_image, right_image, left_points, right_points, draw_matches=True):

    match_image = np.hstack((left_image, right_image))
    left_image_width = left_image.shape[1]
    for left_point, right_point in zip(left_points.astype(int), right_points.astype(int)):
        left_match_point = tuple(left_point)
        right_match_point = tuple(right_point + np.array([left_image_width, 0]))
        cv2.circle(match_image, left_match_point, 5, (0, 255, 0), -1)
        cv2.circle(match_image, right_match_point, 5, (0, 255, 0), -1)
        if draw_matches:
            cv2.line(match_image, left_match_point, right_match_point, (0, 255, 0), 1)

    plt.title('matchings: ' + str(len(left_points)))
    plt.imshow(match_image)
    plt.show()

    return match_image


def plot_epilines(img1, img2, pts1, pts2, F):

    img1 = img1.copy()
    img2 = img2.copy()

    def drawlines(img1, img2, lines, pts1, pts2):
        r, c, _ = img1.shape
        for r, pt1, pt2 in zip(lines, pts1, pts2):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            color = (0, 255, 0)
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
            img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
            img1 = cv2.circle(img1, tuple(pt1.astype(int)), 5, color, -1)
            img2 = cv2.circle(img2, tuple(pt2.astype(int)), 5, color, -1)
        return img1, img2

    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)

    plt.subplot(121), plt.imshow(img5)
    plt.subplot(122), plt.imshow(img3)
    plt.show()
