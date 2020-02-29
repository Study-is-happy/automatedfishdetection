import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
import os

import config

# TODO: Set the dirs

dataset_dir = config.project_dir+'train/'

###########################################################################

images_dir = dataset_dir+'images/'

instances_file_path = dataset_dir+'instances.json'

colors = ['red', 'orange', 'white', 'grey']

with open(instances_file_path) as instances_file:

    instances_dict = json.load(instances_file)

for image_id, instance in instances_dict.items():

    print(image_id)

    if image_id != '20100922.163718.01228':
        continue

    # if len(instance['annotations']) < 20:
    #     continue

    image = mpimg.imread(images_dir+image_id+'.jpg')

    width = instance['width']
    height = instance['height']

    fig = plt.figure(figsize=(20, 20))

    plt.title(image_id)

    plt.imshow(image)

    current_axis = plt.gca()

    instance['annotations'].extend([{
        "category_id": 3,
        "bbox": [
            530./width,
            864./height,
            809./width,
            1037./height
        ]
    }, {
        "category_id": 3,
        "bbox": [
            872./width,
            713./height,
            1044./width,
            879./height
        ]
    }])

    # instance['annotations'].extend([
    #     {
    #         "category_id": 0,
    #         "bbox": [
    #             1123./width,
    #             137./height,
    #             1385./width,
    #             385./height
    #         ],
    #         'score':0.9
    #     },
    #     {
    #         "category_id": 0,
    #         "bbox": [
    #             535./width,
    #             618./height,
    #             848./width,
    #             787./height
    #         ],
    #         'score':0.9
    #     },
    #     {
    #         "category_id": 2,
    #         "bbox": [
    #             1038./width,
    #             264./height,
    #             1802./width,
    #             1120./height
    #         ],
    #         'score':0.9
    #     },
    #     {
    #         "category_id": 0,
    #         "bbox": [
    #             615./width,
    #             413./height,
    #             815./width,
    #             495./height
    #         ],
    #         'score':0.9
    #     },
    #     {
    #         "category_id": 0,
    #         "bbox": [
    #             975./width,
    #             310./height,
    #             1202./width,
    #             448./height
    #         ],
    #         'score':0.9
    #     },
    #     {
    #         "category_id": 2,
    #         "bbox": [
    #             49./width,
    #             1313./height,
    #             829./width,
    #             2044./height
    #         ],
    #         'score':0.9
    #     },
    #     {
    #         "category_id": 0,
    #         "bbox": [
    #             1056./width,
    #             1578./height,
    #             1196./width,
    #             1666./height
    #         ],
    #         'score':0.9
    #     },
    #     {
    #         "category_id": 0,
    #         "bbox": [
    #             1304./width,
    #             1479./height,
    #             1430./width,
    #             1556./height
    #         ],
    #         'score':0.9
    #     },
    #     {
    #         "category_id": 0,
    #         "bbox": [
    #             1061./width,
    #             839./height,
    #             1178./width,
    #             971./height
    #         ],
    #         'score':0.9
    #     },
    #     {
    #         "category_id": 0,
    #         "bbox": [
    #             566./width,
    #             1524./height,
    #             698./width,
    #             1639./height
    #         ],
    #         'score':0.9
    #     },
    #     {
    #         "category_id": 0,
    #         "bbox": [
    #             258./width,
    #             1348./height,
    #             361./width,
    #             1467./height
    #         ],
    #         'score':0.9
    #     },
    #     {
    #         "category_id": 0,
    #         "bbox": [
    #             437./width,
    #             1736./height,
    #             555./width,
    #             1823./height
    #         ],
    #         'score':0.9
    #     },
    #     {
    #         "category_id": 0,
    #         "bbox": [
    #             128./width,
    #             1820./height,
    #             225./width,
    #             1922./height
    #         ],
    #         'score':0.9
    #     }
    # ])

    for annotation in instance['annotations']:

        bbox = annotation['bbox']

        annotation_width = (bbox[2]-bbox[0])*width
        annotation_height = (bbox[3]-bbox[1])*height

        # if (annotation['category_id'] == 0 and (annotation_width < 150 or annotation_height < 150)) or (annotation['category_id'] == 2 and (annotation_width < 710 or annotation_height < 800)):
        #     if 'score' not in annotation:
        #         continue

        # if (annotation['category_id'] == 0 and (annotation_width < 100 or annotation_height < 50)) or (annotation['category_id'] == 2 and (annotation_width < 710 or annotation_height < 700)):
        #     continue
        # annotation['category_id'] = 3

        print(bbox)

        color = colors[annotation['category_id']]

        if 'score' in annotation:
            linestyle = 'dashed'
            # plt.text(bbox[0]*width, bbox[1]*height, format(
            #     annotation['score'], '0.2f'), color=color, size=30)
        else:
            linestyle = '-'

        # if (annotation['category_id'] == 0 and (annotation_width < 100 or annotation_height < 50)) or (annotation['category_id'] == 2 and (annotation_width < 710 or annotation_height < 700) and annotation['category_id'] != 3):
        #     linestyle = 'dashed'
        #     color = 'grey'
        #     annotation['category_id'] = 4

        current_axis.add_patch(plt.Rectangle(
            (bbox[0]*width, bbox[1]*height), (bbox[2]-bbox[0])*width, (bbox[3]-bbox[1])*height, color=color, fill=False, linewidth=3, linestyle=linestyle))

        plt.text(bbox[0]*width, bbox[1]*height-3,
                 config.categories[annotation['category_id']], color='white', size=30, bbox={'facecolor': color, 'alpha': 0.5, 'pad': 3})

    plt.show()
