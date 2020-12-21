import csv
import json
import os
import numpy as np

import util
import config

results_path = config.project_dir + 'results/' + config.results_name

results_approve_path = config.project_dir + \
    'results_approve/' + config.results_name

abs_timer_threshold = 10
approve_confident_count = 6


def calc_timer(edge_timer, corner_timer):
    return edge_timer + corner_timer * 1.5


print_results = {'approve': 0, 'reject': 0, 'empty': 0, 'bug': 0}

with open(config.project_dir + 'easy_gt/instances.json') as easy_gt_instances_file:
    easy_gt_instances = json.load(easy_gt_instances_file)

with open(results_path) as results_file:

    results = csv.reader(results_file)

    headers = next(results)

    headers.append('conf_indexes')
    headers.append('not_conf_indexes')
    headers.append('reject_indexes')
    headers.append('gt_indexes')

    with open(results_approve_path, 'w') as results_approve_file:

        writer = csv.writer(results_approve_file, quoting=csv.QUOTE_NONNUMERIC)

        writer.writerow(headers)

        for result in results:

            result.extend([''] * (len(headers) - len(result)))

            result_annotations = json.loads(result[-8])

            with open(config.project_dir + 'predict/annotations/' + result[-9] + '.json') as predict_annotations_file:
                predict_annotations = json.load(predict_annotations_file)

            conf_indexes = []
            not_conf_indexes = []
            reject_indexes = []

            reject_reasons = set()

            annotation_indexes = []
            gt_indexes = []

            for index, predict_annotation in enumerate(predict_annotations):
                if 'gt_annotation_index' in predict_annotation:
                    gt_indexes.append(index)
                else:
                    annotation_indexes.append(index)

            result[-1] = gt_indexes

            bug = False

            if len(result_annotations) == len(predict_annotations):

                approve = True

                for gt_index in gt_indexes:

                    predict_annotation = predict_annotations[gt_index]
                    easy_gt_annotation = easy_gt_instances[predict_annotation['image_id']
                                                           ]['annotations'][predict_annotation['gt_annotation_index']]

                    result_annotation = result_annotations[gt_index]

                    if None in result_annotation['bbox']:
                        approve = False
                        bug = True
                        continue

                    if easy_gt_annotation['category_id'] != result_annotation['category_id']:
                        approve = False
                        reject_reasons.add('Wrong species')

                    if util.get_bboxes_iou(easy_gt_annotation['bbox'], result_annotation['bbox']) < easy_gt_annotation['iou']:
                        approve = False
                        reject_reasons.add(
                            'Bounding box not fitting tightly')

                    gt_timer = calc_timer(
                        result_annotation['edge_timer'], result_annotation['corner_timer'])

                if approve:

                    confident_count = 0

                    for annotation_index in annotation_indexes:

                        predict_annotation = predict_annotations[annotation_index]
                        result_annotation = result_annotations[annotation_index]

                        if None in result_annotation['bbox']:
                            bug = True
                            reject_indexes.append(annotation_index)
                            continue

                        confident = True

                        if config.categories[result_annotation['category_id']] != 'background':

                            if calc_timer(result_annotation['edge_timer'], result_annotation['corner_timer']) < min(abs_timer_threshold, gt_timer):
                                confident = False
                                reject_reasons.add('Bad bounding box')

                            target_iou = util.get_bboxes_iou(
                                predict_annotation['bbox'], result_annotation['bbox'])

                            if target_iou == 0:
                                confident = False
                                reject_reasons.add(
                                    'Labeling unwanted object')

                            exist_annotations_file_path = config.project_dir + \
                                'predict/exist_annotations/' + \
                                result_annotation['image_id'] + '.json'

                            if os.path.exists(exist_annotations_file_path):

                                with open(exist_annotations_file_path) as exist_annotations_file:
                                    exist_annotations = json.load(
                                        exist_annotations_file)

                                for exist_annotation in exist_annotations:
                                    if util.get_bboxes_iou(exist_annotation['bbox'], result_annotation['bbox']) > target_iou:
                                        confident = False
                                        reject_reasons.add(
                                            'Duplicate bounding box')
                                        break

                        if confident:
                            confident_count += 1
                            if predict_annotation['category_id'] == result_annotation['category_id']:
                                conf_indexes.append(annotation_index)
                            else:
                                not_conf_indexes.append(annotation_index)
                        else:
                            reject_indexes.append(annotation_index)

                    if confident_count >= approve_confident_count:
                        result[-6] = 'x'
                        print_results['approve'] += 1

                    else:
                        approve = False
                        conf_indexes = []
                        not_conf_indexes = []

                if bug:
                    result[-6] = 'x'
                    print_results['bug'] += 1

                if not approve:
                    result[-5] = ', '.join(reject_reasons)
                    reject_indexes = annotation_indexes
                    print_results['reject'] += 1

                result[-4] = conf_indexes
                result[-3] = not_conf_indexes
                result[-2] = reject_indexes

            else:
                result[-5] = 'Getting empty results, might be something wrong with MTurk'
                result[-4] = []
                result[-3] = []
                result[-2] = annotation_indexes
                print_results['empty'] += 1

            # if bug:
            #     print(result_annotations)
            writer.writerow(result)

print(print_results)
