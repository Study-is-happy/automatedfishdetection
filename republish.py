import json
import csv
import os
import shutil

import config
import util

results_approve_path = config.project_dir + \
    'results_approve/' + config.results_name

exist_annotations_dir = config.project_dir + 'predict/exist_annotations/'
shutil.rmtree(exist_annotations_dir)
os.mkdir(exist_annotations_dir)

with open(config.project_dir + 'predict/annotations/cache.json') as cache_annotations_file:
    cache_annotations = json.load(cache_annotations_file)

with open(config.project_dir + 'update/instances.json') as update_instances_file:
    update_instances = json.load(update_instances_file)

with open(config.project_dir + 'easy_gt/instances.json') as easy_gt_instances_file:
    easy_gt_annotation_generator = util.easy_gt_annotation_generator(
        json.load(easy_gt_instances_file))

easy_gt_index_generator = util.easy_gt_index_generator()

with open(config.project_dir + 'predict/annotation_ids.csv') as annotation_ids_file:
    annotation_id = int(annotation_ids_file.readlines()[-1]) + 1

start_annotation_id = annotation_id

with open(results_approve_path) as results_approve_file:

    results = csv.reader(results_approve_file)

    headers = next(results)

    for result in results:

        result_annotations = json.loads(result[-8])

        predict_annotations_file_path = config.project_dir + \
            'predict/annotations/' + result[-9] + '.json'

        with open(predict_annotations_file_path) as predict_annotations_file:
            predict_annotations = json.load(predict_annotations_file)

        not_conf_indexes = json.loads(result[-3])
        reject_indexes = json.loads(result[-2])

        for index in not_conf_indexes:
            predict_annotations[index]['category_id'] = result_annotations[index]['category_id']

        for index in not_conf_indexes + reject_indexes:
            cache_annotations.append(predict_annotations[index])

        while len(cache_annotations) >= config.predict_per_file:

            current_annotations = cache_annotations[:config.predict_per_file]

            for current_annotation in current_annotations:
                image_id = current_annotation['image_id']
                if image_id in update_instances:

                    exist_annotations_file_path = exist_annotations_dir + image_id + '.json'

                    if not os.path.exists(exist_annotations_file_path):
                        util.write_json_file(
                            update_instances[image_id]['annotations'], exist_annotations_file_path)
                else:
                    print(image_id)

            easy_gt_index = next(easy_gt_index_generator)
            easy_gt_annotation = next(easy_gt_annotation_generator)
            current_annotations.insert(
                easy_gt_index, easy_gt_annotation)

            annotations_file_path = config.project_dir + 'predict/annotations/' + str(annotation_id) + '.json'

            util.write_json_file(
                current_annotations, annotations_file_path)

            shutil.copy(annotations_file_path,
                        config.project_dir + 'predict/current_annotations/')

            cache_annotations = cache_annotations[config.predict_per_file:]

            annotation_id += 1

util.write_json_file(
    cache_annotations, config.project_dir + 'predict/annotations/cache.json')

with open(config.project_dir + 'predict/annotation_ids.csv', 'w') as annotations_csv_file:

    annotations_csv_file.write('annotation_id\n')
    for i in range(start_annotation_id, annotation_id):
        annotations_csv_file.write(str(i) + '\n')
