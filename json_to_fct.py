import json
import os
import glob

import utils
import config

with open(config.project_dir + 'all/instances_1.json') as instances_file:

    instances = json.load(instances_file)

    for image_id, instance in instances.items():

        width = instance['width']
        height = instance['height']

        predict_annotations = []

        for annotation in instance['annotations']:

            if annotation['category_id'] == 9:

                predict_annotations.append(annotation)

        if len(predict_annotations) > 0:

            fct_file_path = glob.glob('/media/auv/Seagate Desktop Drive/AUV_images_fcts/*/*/port/port_rectified/' + image_id + '.fct')[0]

            with open(fct_file_path) as prev_fct_file:

                with open(config.project_dir + 'fct/' + image_id + '.fct', 'w') as fct_file:

                    prev_annotation_count = 0

                    for annotation_line in prev_fct_file.readlines():
                        fct_file.write(annotation_line)
                        annotation_line = annotation_line.strip().split(',')
                        if len(annotation_line) > 1:
                            prev_annotation_count += 1
                            annotation_line_template = annotation_line[:18].copy()

                    annotation_line_template[15] = 'NaN'
                    annotation_line_template[16] = 'NaN'
                    annotation_line_template[17] = 'NaN'

                    for annotation_index, predict_annotation in enumerate(predict_annotations):
                        annotation_line = annotation_line_template.copy()
                        annotation_line[10] = 'Rockfish'
                        annotation_line[11] = 'predict_rockfish'
                        annotation_line[12] = 'RO' + str(prev_annotation_count + annotation_index + 1)
                        bbox = predict_annotation['bbox']
                        utils.rel_to_abs(bbox, width, height)
                        center_x = (bbox[0] + bbox[2]) / 2
                        center_y = height - (bbox[1] + bbox[3]) / 2
                        annotation_line[13] = str(center_x)
                        annotation_line[14] = str(center_y)
                        fct_file.write(','.join(annotation_line) + '\n')
