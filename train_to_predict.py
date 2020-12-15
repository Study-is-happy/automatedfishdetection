import json
import shutil

import config
import util
import reset_predict

shutil.copy(config.project_dir+'train/instances.json',
            config.project_dir+'predict/')

with open(config.project_dir+'train/instances.json') as train_instances_file:
    train_instances = json.load(train_instances_file)

with open(config.project_dir+'easy_gt/instances.json') as easy_gt_instances_file:
    easy_gt_annotation_generator = util.easy_gt_annotation_generator(
        json.load(easy_gt_instances_file))

easy_gt_index_generator = util.easy_gt_index_generator()

cache_annotations = []

annotation_id = 0

for image_id, instance in train_instances.items():
    width = instance['width']
    height = instance['height']
    for annotation in instance['annotations']:
        cache_annotations.append({'image_id': image_id, 'width': width, 'height': height, 'category_id': annotation['category_id'], 'score': 1,
                                  'bbox': annotation['bbox']})

    shutil.copy(config.project_dir+'train/images/'+image_id+'.jpg',
                config.project_dir+'predict/images/')

    while len(cache_annotations) >= config.predict_per_file:

        current_annotations = cache_annotations[:config.predict_per_file]

        easy_gt_index = next(easy_gt_index_generator)
        easy_gt_annotation = next(easy_gt_annotation_generator)
        shutil.copy(config.project_dir+'easy_gt/images/'+easy_gt_annotation['image_id']+'.jpg',
                    config.project_dir+'predict/images/')
        current_annotations.insert(
            easy_gt_index, easy_gt_annotation)

        util.write_json_file(
            current_annotations, config.project_dir+'predict/annotations/'+str(annotation_id)+'.json')

        cache_annotations = cache_annotations[config.predict_per_file:]

        annotation_id += 1

util.write_json_file(
    cache_annotations, config.project_dir+'predict/annotations/cache.json')

with open(config.project_dir+'predict/annotation_ids.csv', 'a') as annotation_ids_file:

    for i in range(annotation_id):
        annotation_ids_file.write(str(i) + '\n')
