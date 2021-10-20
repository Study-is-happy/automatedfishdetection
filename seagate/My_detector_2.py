import numpy as np
import cv2

import utils


class My_detector:
    def __init__(self):
        self.

        self.brief = cv2.xfeatures2d.BriefDescriptorExtractor_create(64)
        self.brief_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (57, 57))

    def detectAndCompute(self, image, roi=None):

        image_shape = np.array(image.shape)

        image_list = [image]

        scale_list = [np.array([1.0, 1.0])]

        for level in range(1, self.levels):

            current_image = cv2.resize(image_list[-1], (0, 0), fx=self.ratio, fy=self.ratio)

            image_list.append(current_image)

            scale_list.append(image_shape / np.array(current_image.shape))

        response_list = []

        scale_ys_list = []
        scale_xs_list = []

        ys_list = []
        xs_list = []

        level_list = []

        for level, (image, scale) in enumerate(zip(image_list, scale_list)):

            eigen_image = cv2.cornerMinEigenVal(image, 2, 3)

            # test_eigen_image = cv2.cornerEigenValsAndVecs(image, 2, 3)[:, :, :2]
            # test_eigen_image = np.amax(test_eigen_image, axis=-1)

            threshold_image = np.logical_and(cv2.dilate(eigen_image, self.non_max_kernel) == eigen_image, eigen_image > 0)

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

        feature_mask = np.zeros(len(response_list), bool)

        for row in range(self.num_rows):

            ys_mask = np.logical_and(ys_list >= row * section_row, ys_list < (row + 1) * section_row)

            for col in range(self.num_cols):

                xs_mask = np.logical_and(xs_list >= col * section_col, xs_list < (col + 1) * section_col)

                section_mask = np.logical_and(ys_mask, xs_mask)

                top_mask = np.flip(np.argsort(response_list[section_mask]))[:self.num_features]

                feature_mask[[sub_mask[top_mask] for sub_mask in np.nonzero(section_mask)]] = True

        response_list = response_list[feature_mask]

        scale_ys_list = scale_ys_list[feature_mask]
        scale_xs_list = scale_xs_list[feature_mask]

        ys_list = ys_list[feature_mask]
        xs_list = xs_list[feature_mask]

        level_list = level_list[feature_mask]

        if roi is None:
            roi = np.full(image_shape, 255, np.uint8)

        roi = cv2.copyMakeBorder(roi, 1, 1, 1, 1, cv2.BORDER_CONSTANT)
        roi = cv2.erode(roi, self.brief_kernel)
        roi = roi[1:roi.shape[0] - 1, 1:roi.shape[1] - 1]

        roi_mask = roi[utils.get_rint(ys_list), utils.get_rint(xs_list)] != 0

        response_list = response_list[roi_mask]

        scale_ys_list = scale_ys_list[roi_mask]
        scale_xs_list = scale_xs_list[roi_mask]

        ys_list = ys_list[roi_mask]
        xs_list = xs_list[roi_mask]

        level_list = level_list[roi_mask]

        keypoints = []
        descriptors = []

        for level in range(self.levels):

            level_mask = level_list == level

            level_response_list = response_list[level_mask]

            level_scale_ys_list = scale_ys_list[level_mask]
            level_scale_xs_list = scale_xs_list[level_mask]

            level_ys_list = ys_list[level_mask]
            level_xs_list = xs_list[level_mask]

            level_keypoints = [cv2.KeyPoint(x, y, 0, response=response) for y, x, response in zip(level_ys_list, level_xs_list, level_response_list)]

            level_keypoints, level_descriptors = self.brief.compute(image_list[level], level_keypoints)

            if len(level_keypoints) > 0:
                keypoints.extend(level_keypoints)
                descriptors.extend(level_descriptors)

        return keypoints, descriptors
