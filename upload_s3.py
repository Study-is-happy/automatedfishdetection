import boto3
import os

import config

# TODO: Set the dir

# upload_dir = 'images/'
# upload_dir = 'current_annotations/'
upload_dir = 'exist_annotations/'

###########################################################################

s3_bucket_dir = 'zhiyongzhang/predict/' + upload_dir
local_dir = config.project_dir + 'predict/' + upload_dir

s3_bucket = boto3.resource('s3',
                           aws_access_key_id=config.aws_access_key_id,
                           aws_secret_access_key=config.aws_secret_access_key).Bucket('datasetocean')

s3_bucket.objects.filter(Prefix=s3_bucket_dir).delete()


for file_path in os.listdir(local_dir):

    s3_bucket.upload_file(local_dir + file_path, s3_bucket_dir +
                          file_path, ExtraArgs={'ACL': 'public-read'})
