import numpy as np
import cv2
import matplotlib.pyplot as plt

import utils


class My_detector:
    def __init__(self, num_features=20, num_rows=41, num_cols=48, levels=6, ratio=0.75):
        self.num_features = num_features
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.levels = levels
        self.ratio = ratio
        self.non_max_kernel = np.ones((5, 5), np.uint8)

        self.brief = cv2.xfeatures2d.BriefDescriptorExtractor_create(64, use_orientation=False)
        self.brief_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (57, 57))

    def detectAndCompute(self, image, mask=None):

        image_shape = np.array(image.shape)

        image_list = [image]

        if mask is None:
            mask = np.full(image_shape, 255, np.uint8)
        mask_list = [mask]

        scale_list = [np.array([1.0, 1.0])]

        for level in range(1, self.levels):

            image_list.append(cv2.resize(image_list[-1], (0, 0), fx=self.ratio, fy=self.ratio))
            mask_list.append(cv2.resize(mask_list[-1], (0, 0), fx=self.ratio, fy=self.ratio))

            scale_list.append(image_shape / np.array(image_list[-1].shape))

        response_list = []

        scale_ys_list = []
        scale_xs_list = []

        ys_list = []
        xs_list = []

        level_list = []

        for level, (image, mask, scale) in enumerate(zip(image_list, mask_list, scale_list)):

            eigen_image = cv2.cornerMinEigenVal(image, 2, 3)

            # test_eigen_image = cv2.cornerEigenValsAndVecs(image, 2, 3)[:, :, :2]
            # test_eigen_image = np.amax(test_eigen_image, axis=-1)

            threshold_image = np.logical_and(cv2.dilate(eigen_image, self.non_max_kernel) == eigen_image, eigen_image > 0)

            mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT)
            mask = cv2.erode(mask, self.brief_kernel)
            mask = mask[1:mask.shape[0] - 1, 1:mask.shape[1] - 1]

            threshold_image[mask == 0] = False

            scale_ys, scale_xs = np.nonzero(threshold_image)

            response_list.extend(eigen_image[scale_ys, scale_xs])

            scale_ys_list.extend(scale_ys)
            scale_xs_list.extend(scale_xs)

            ys = scale_ys * scale[0] + (scale[0] - 1) / 2
            xs = scale_xs * scale[1] + (scale[1] - 1) / 2

            ys_list.extend(ys)
            xs_list.extend(xs)

            level_list.extend(np.full(len(scale_ys), level))

        response_list = np.array(response_list)

        scale_ys_list = np.array(scale_ys_list)
        scale_xs_list = np.array(scale_xs_list)

        ys_list = np.array(ys_list)
        xs_list = np.array(xs_list)

        level_list = np.array(level_list)

        section_row = image_shape[0] / self.num_rows
        section_col = image_shape[1] / self.num_cols

        tile_mask = np.zeros(len(response_list), bool)

        for row in range(self.num_rows):

            ys_mask = np.logical_and(ys_list >= row * section_row, ys_list < (row + 1) * section_row)

            for col in range(self.num_cols):

                xs_mask = np.logical_and(xs_list >= col * section_col, xs_list < (col + 1) * section_col)

                section_mask = np.logical_and(ys_mask, xs_mask)

                top_mask = np.flip(np.argsort(response_list[section_mask]))[:self.num_features]

                tile_mask[[sub_mask[top_mask] for sub_mask in np.nonzero(section_mask)]] = True

        response_list = response_list[tile_mask]

        scale_ys_list = scale_ys_list[tile_mask]
        scale_xs_list = scale_xs_list[tile_mask]

        ys_list = ys_list[tile_mask]
        xs_list = xs_list[tile_mask]

        level_list = level_list[tile_mask]

        keypoints_list = []
        descriptors_list = []

        for level in range(self.levels):

            level_mask = level_list == level

            level_image = image_list[level]

            level_response_list = response_list[level_mask]

            level_scale_ys_list = scale_ys_list[level_mask]
            level_scale_xs_list = scale_xs_list[level_mask]

            level_ys_list = ys_list[level_mask]
            level_xs_list = xs_list[level_mask]

            level_scale_keypoints = [cv2.KeyPoint(float(scale_x), float(scale_y), 0, response=response) for scale_y, scale_x, response in zip(level_scale_ys_list, level_scale_xs_list, level_response_list)]

            _, level_descriptors = self.brief.compute(level_image, level_scale_keypoints)

            level_keypoints = [cv2.KeyPoint(x, y, 0, response=response) for y, x, response in zip(level_ys_list, level_xs_list, level_response_list)]

            if len(level_keypoints) > 0:
                keypoints_list.append(level_keypoints)
                descriptors_list.append(level_descriptors)

        return keypoints_list, descriptors_list
