import csv
import json
import os
import numpy as np

import util
import config

results_path = config.project_dir+'results/' + config.results_name

results_approve_path = config.project_dir + \
    'results_approve/' + config.results_name

iou_threshold = 0.7
abs_timer_threshold = 15
approve_rate = 0.7


def calc_timer(edge_timer, corner_timer):
    return edge_timer + corner_timer*1.5


print_results = {'approve': 0, 'reject': 0, 'empty': 0}

with open(results_path) as results_file:

    results = csv.reader(results_file)

    headers = next(results)

    headers.append('conf_indexes')
    headers.append('not_conf_indexes')
    headers.append('approved_gt_indexes')
    headers.append('reject_indexes')

    with open(results_approve_path, 'w') as results_approve_file:

        writer = csv.writer(results_approve_file, quoting=csv.QUOTE_NONNUMERIC)

        writer.writerow(headers)

        for result in results:

            result.extend(['']*(len(headers)-len(result)))

            result_annotations = json.loads(result[-8])

            with open(config.project_dir+'predict/annotations/'+result[-9]+'.json') as predict_annotations_file:
                predict_annotations = json.load(predict_annotations_file)

            conf_indexes = []
            not_conf_indexes = []
            approved_gt_indexes = []
            reject_indexes = []

            unchecked_indexes = []

            reject_reasons = set()

            result_length = len(result_annotations)

            if result_length == config.annotation_per_file:

                for index in range(result_length):

                    if index in config.gt_indexes:

                        gt_annotation = predict_annotations[index]
                        result_annotation = result_annotations[index]

                        approve = True

                        if gt_annotation['category_id'] != result_annotation['category_id']:
                            approve = False
                            reject_reasons.add('Wrong species')

                        if gt_annotation['image_id'] == '20161027.175242.00310_rect_color':
                            if util.get_bboxes_iou(gt_annotation['bbox'], result_annotation['bbox']) < 0.53:
                                approve = False
                                reject_reasons.add(
                                    'Bounding box not fitting tightly')

                        elif util.get_bboxes_iou(gt_annotation['bbox'], result_annotation['bbox']) < iou_threshold:
                            approve = False
                            reject_reasons.add(
                                'Bounding box not fitting tightly')

                        if approve:

                            approved_gt_indexes.append(index)

                            gt_timer = calc_timer(
                                result_annotation['edge_timer'], result_annotation['corner_timer'])

                            for unchecked_index in unchecked_indexes:
                                predict_annotation = predict_annotations[unchecked_index]
                                result_annotation = result_annotations[unchecked_index]

                                approve = True

                                if result_annotation['category_id'] != len(config.categories)-1:

                                    result_timer = calc_timer(
                                        result_annotation['edge_timer'], result_annotation['corner_timer'])
                                    if result_timer < min(abs_timer_threshold, gt_timer):
                                        approve = False
                                        reject_reasons.add('Bad bounding box')

                                    exist_annotations_file_path = config.project_dir + \
                                        'predict/exist_annotations/' + \
                                        result_annotation['image_id']+'.json'

                                    if os.path.exists(exist_annotations_file_path):

                                        with open(exist_annotations_file_path) as exist_annotations_file:
                                            exist_annotations = json.load(
                                                exist_annotations_file)

                                        for exist_annotation in exist_annotations:
                                            if util.get_bboxes_iou(exist_annotation['bbox'], result_annotation['bbox']) > 0.7:
                                                approve = False
                                                reject_reasons.add(
                                                    'Duplicate bounding box')
                                                break

                                if approve:
                                    if predict_annotation['category_id'] == result_annotation['category_id'] and util.get_bboxes_iou(predict_annotation['bbox'], result_annotation['bbox']) > 0:
                                        conf_indexes.append(unchecked_index)
                                    else:
                                        not_conf_indexes.append(
                                            unchecked_index)
                                else:
                                    reject_indexes.append(unchecked_index)

                        else:
                            reject_indexes.extend(unchecked_indexes)

                        unchecked_indexes = []

                    else:
                        unchecked_indexes.append(index)

                if len(conf_indexes)+len(not_conf_indexes)+len(approved_gt_indexes) >= result_length*approve_rate:
                    result[-6] = 'x'
                    print_results['approve'] += 1

                else:
                    result[-5] = ', '.join(reject_reasons)
                    reject_indexes += conf_indexes+not_conf_indexes
                    conf_indexes = []
                    not_conf_indexes = []
                    print_results['reject'] += 1

                result[-4] = conf_indexes
                result[-3] = not_conf_indexes
                result[-2] = approved_gt_indexes
                result[-1] = reject_indexes

            else:
                result[-5] = 'Getting empty results, might be something wrong with MTurk'
                result[-4] = []
                result[-3] = []
                result[-2] = []
                all_indexes = list(range(config.annotation_per_file))
                for gt_index in config.gt_indexes:
                    all_indexes.remove(gt_index)
                result[-1] = all_indexes
                print_results['empty'] += 1

            writer.writerow(result)

print(print_results)
