import os
import config

count = 0

train_file_names = os.listdir(config.project_dir+'train/images/')

update_images_dir = config.project_dir+'update/images/'
for update_file_name in os.listdir(update_images_dir):
    if update_file_name in train_file_names:
        os.remove(update_images_dir+update_file_name)
