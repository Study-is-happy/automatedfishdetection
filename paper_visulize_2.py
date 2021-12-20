import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
import os

import config

dataset_dir = config.project_dir + 'update/'

images_dir = dataset_dir + 'images/'

instances_file_name_list = ['instances_pro_1.json', 'instances_pro_2.json',
                            'instances_pro_3.json', 'instances_pro_4.json',
                            'instances_pro_5.json', 'instances_pro_6.json']

colors = ['red', 'orange', 'white', 'grey']

instances_dict_list = []

for instances_file_name in instances_file_name_list:

    with open(dataset_dir + instances_file_name) as instances_file:

        instances_dict_list.append(json.load(instances_file))

len_annotations_dict = {}

for image_id, instance in instances_dict_list[0].items():
    len_annotations_dict[image_id] = len(instance['annotations'])

best_len_annotations_diff = 0
best_image_id = None

used_image_id_list = ['20100920.224052.01124', '20100922.160429.00354', '20100922.160434.00356', '20100922.155950.00230']

for image_id, instance in instances_dict_list[-1].items():

    if image_id in used_image_id_list:
        continue

    if image_id in len_annotations_dict:
        len_annotations_diff = len(instance['annotations']) - len_annotations_dict[image_id]
    else:
        len_annotations_diff = len(instance['annotations'])

    if len_annotations_diff > best_len_annotations_diff:
        best_len_annotations_diff = len_annotations_diff
        best_image_id = image_id

print(len(instances_dict_list[0][best_image_id]['annotations']))
print(len(instances_dict_list[-1][best_image_id]['annotations']))

print(best_image_id)

prev_len_annotations = 0
vis_instances_dict_list = []

for instances_dict in instances_dict_list:
    len_annotations = len(instances_dict[best_image_id]['annotations'])

    # print(len_annotations)

    if len_annotations - prev_len_annotations >= 3:
        prev_len_annotations = len_annotations
        vis_instances_dict_list.append(instances_dict)
        print(len_annotations)

print(len_annotations)

vis_instances_dict_list[-1] = instances_dict_list[-1]

for vis_instances_dict in vis_instances_dict_list:

    instance = vis_instances_dict[best_image_id]

    image = mpimg.imread(images_dir + best_image_id + '.jpg')

    width = instance['width']
    height = instance['height']

    fig = plt.figure(figsize=(20, 20))

    plt.title(best_image_id)

    plt.imshow(image)
    # plt.show()

    current_axis = plt.gca()

    prev_len_annotations = len_annotations

    for annotation in instance['annotations']:

        bbox = annotation['bbox']

        color = config.colors[annotation['category_id']]

        current_axis.add_patch(plt.Rectangle(
            (bbox[0] * width, bbox[1] * height), (bbox[2] - bbox[0]) * width, (bbox[3] - bbox[1]) * height, color=color, fill=False, linewidth=3))

        plt.text(bbox[0] * width, bbox[1] * height - 3,
                 config.categories[annotation['category_id']], color='white', size=10, bbox={'facecolor': color, 'alpha': 0.5, 'pad': 3})

    plt.show()
