import json
import csv
import os

import config
import util

results_approve_path = config.project_dir + \
    'results_approve/' + config.results_name

with open(config.project_dir+'predict/annotations/cache.json') as cache_annotations_file:
    cache_annotations = json.load(cache_annotations_file)

with open(config.project_dir+'easy_gt/instances.json') as easy_gt_instances_file:
    easy_gt_annotation_generator = util.easy_gt_annotation_generator(
        json.load(easy_gt_instances_file))

with open(config.project_dir+'predict/annotation_ids.csv') as annotation_ids_file:
    annotation_id = int(annotation_ids_file.readlines()[-1])+1

start_annotation_id = annotation_id

with open(results_approve_path) as results_approve_file:

    results = csv.reader(results_approve_file)

    headers = next(results)

    for result in results:

        result_annotations = json.loads(result[-8])

        predict_annotations_file_path = config.project_dir + \
            'predict/annotations/'+result[-9]+'.json'

        with open(predict_annotations_file_path) as predict_annotations_file:
            predict_annotations = json.load(predict_annotations_file)

        not_conf_indexes = json.loads(result[-3])
        reject_indexes = json.loads(result[-2])

        for index in not_conf_indexes:
            predict_annotations[index]['category_id'] = result_annotations[index]['category_id']

        for index in not_conf_indexes+reject_indexes:
            cache_annotations.append(predict_annotations[index])

        while len(cache_annotations) >= config.predict_per_file:

            current_annotations = cache_annotations[:config.predict_per_file]

            for gt_index in config.gt_indexes:
                easy_gt_annotation = next(easy_gt_annotation_generator)
                current_annotations.insert(
                    gt_index, easy_gt_annotation)

            util.write_json_file(
                current_annotations, config.project_dir+'predict/annotations/'+str(annotation_id)+'.json')

            cache_annotations = cache_annotations[config.predict_per_file:]

            annotation_id += 1

util.write_json_file(
    cache_annotations, config.project_dir+'predict/annotations/cache.json')

with open(config.project_dir+'predict/annotation_ids.csv', 'w') as annotations_csv_file:

    annotations_csv_file.write('annotation_id\n')
    for i in range(start_annotation_id, annotation_id):
        annotations_csv_file.write(str(i) + '\n')
