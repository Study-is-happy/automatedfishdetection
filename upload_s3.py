import boto3
import os

import config

s3_bucket = boto3.resource('s3',
                           aws_access_key_id=config.aws_access_key_id,
                           aws_secret_access_key=config.aws_secret_access_key).Bucket('datasetocean')

s3_bucket.objects.filter(Prefix="zhiyongzhang/predict/").delete()

for root, dirs, files in os.walk(config.project_dir+'predict/'):
    for file in files:
        full_file_path = os.path.join(root, file)
        s3_bucket.upload_file(full_file_path, 'zhiyongzhang/' + os.path.relpath(
            full_file_path, config.project_dir), ExtraArgs={'ACL': 'public-read'})


# print(len(s3_bucket.objects.filter(Prefix="zhiyongzhang/predict/")))
