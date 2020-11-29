import shutil

import config
import util

# TODO: Set the dirs

src_dataset_dir = config.project_dir+'pmfs (copy)/'

src_images_dir = src_dataset_dir+'images/'

src_annotations_dir = src_dataset_dir+'annotations/'

src_images_file_path = src_dataset_dir+'images.txt'

des_dataset_dir = config.project_dir+'pmfs/train/'

des_images_dir = des_dataset_dir+'images/'

des_annotations_dir = des_dataset_dir+'annotations/'

###########################################################################

shutil.copy(src_images_file_path, des_dataset_dir)

with open(src_images_file_path) as src_images_file:
    for image_id in src_images_file:

        image_id = image_id.rstrip('\n')

        shutil.copy(src_images_dir+image_id+'.jpg', des_images_dir)
        shutil.copy(src_annotations_dir+image_id+'.xml', des_annotations_dir)
