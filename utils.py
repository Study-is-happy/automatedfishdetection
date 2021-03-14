import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt


def write_json_file(obj, json_file_path):

    with open(json_file_path, 'w') as json_file:

        json.dump(obj, json_file, indent='\t')


def get_bboxes_iou(bbox1, bbox2):

    width1 = bbox1[2] - bbox1[0]

    width2 = bbox2[2] - bbox2[0]

    overlap_width = width1 + width2 - (max(bbox1[2], bbox2[2])
                                       - min(bbox1[0], bbox2[0]))

    height1 = bbox1[3] - bbox1[1]

    height2 = bbox2[3] - bbox2[1]

    overlap_height = height1 + height2 - (max(bbox1[3], bbox2[3])
                                          - min(bbox1[1], bbox2[1]))

    if overlap_width > 0 and overlap_height > 0:

        area1 = width1 * height1

        overlap_area = overlap_width * overlap_height

        area2 = width2 * height2

        iou = float(overlap_area) / \
            (area1 + area2 - overlap_area)

        return iou

    return 0


def get_bbox_size(bbox, image_width, image_height):
    return (bbox[2] - bbox[0]) * image_width * (bbox[3] - bbox[1]) * image_height


def filter_overlap_instance(instance, iou):

    annotations = instance['annotations']

    width = instance['width']
    height = instance['height']

    for index, annotation in enumerate(annotations):

        for unchecked_annotation in annotations[index + 1:]:
            if get_bboxes_iou(annotation['bbox'], unchecked_annotation['bbox']) > iou:
                if get_bbox_size(annotation['bbox'], width, height) < get_bbox_size(unchecked_annotation['bbox'], width, height):
                    unchecked_annotation['overlap'] = True
                else:
                    annotation['overlap'] = True

    instance['annotations'] = [annotation for annotation in annotations if 'overlap' not in annotation]

    return [annotation for annotation in annotations if 'overlap' in annotation]


def abs_to_rel(bbox, image_width, image_height):

    bbox[0] /= image_width
    bbox[1] /= image_height
    bbox[2] /= image_width
    bbox[3] /= image_height


def rel_to_abs(bbox, image_width, image_height):

    bbox[0] = bbox[0] * image_width
    bbox[1] = bbox[1] * image_height
    bbox[2] = bbox[2] * image_width
    bbox[3] = bbox[3] * image_height


def pascal_voc_abs_to_rel(bbox, image_width, image_height):

    bbox[0] = (bbox[0] - 1) / image_width
    bbox[1] = (bbox[1] - 1) / image_height
    bbox[2] /= image_width
    bbox[3] /= image_height


def pascal_voc_rel_to_abs(bbox, image_width, image_height):

    bbox[0] = bbox[0] * image_width + 1
    bbox[1] = bbox[1] * image_height + 1
    bbox[2] = bbox[2] * image_width
    bbox[3] = bbox[3] * image_height


def norm_rel_bbox(bbox):

    for i in range(4):
        if bbox[i] < 0:
            bbox[i] = 0
        elif bbox[i] > 1:
            bbox[i] = 1


def get_rint(num):
    return np.rint(num).astype(int)


def get_non_max_suppression_mask(keypoints, image_shape):
    binary_image = np.zeros(image_shape)
    response_list = np.array([keypoint.response for keypoint in keypoints])
    mask = np.flip(np.argsort(response_list))
    point_list = get_rint([keypoint.pt for keypoint in keypoints])[mask]
    non_max_suppression_mask = []
    for point, index in zip(point_list, mask):
        if binary_image[point[1], point[0]] == 0:
            non_max_suppression_mask.append(index)
            cv2.circle(binary_image, (point[0], point[1]), 2, 255, -1)

    return non_max_suppression_mask


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

    plt.imshow(match_image)
    plt.show()


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


def easy_gt_index_generator():
    while True:
        for index in range(5, 10):
            yield index


def easy_gt_annotation_generator(instances):
    while True:
        for image_id, instance in instances.items():
            for index, annotation in enumerate(instance['annotations']):
                if annotation['iou'] > 0.5:
                    yield {'image_id': image_id, 'width': instance['width'], 'height': instance['height'], 'category_id': annotation['category_id'], 'gt_annotation_index': index, 'score': 1,
                           'bbox': annotation['bbox']}
