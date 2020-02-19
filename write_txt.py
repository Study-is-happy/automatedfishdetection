import os

import config

dataset_dir = config.project_dir+'train/'

with open(dataset_dir+'images.txt', 'w') as image_ids_file:
    for annotation_file_name in os.listdir(dataset_dir+'annotations/'):
        image_id = os.path.splitext(annotation_file_name)[0]
        image_ids_file.write(image_id + '\n')
