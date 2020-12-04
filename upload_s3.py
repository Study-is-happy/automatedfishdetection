import boto3
import os

import config

resource = boto3.resource('s3',
                          aws_access_key_id=config.aws_access_key_id,
                          aws_secret_access_key=config.aws_secret_access_key)

bucket = resource.Bucket('datasetocean')
bucket.objects.filter(Prefix="zhiyongzhang/predict/").delete()

client = boto3.client('s3',
                      aws_access_key_id=config.aws_access_key_id,
                      aws_secret_access_key=config.aws_secret_access_key)

for root, dirs, files in os.walk(config.project_dir+'predict/'):
    for file in files:
        full_file_path = os.path.join(root, file)
        client.upload_file(full_file_path, 'datasetocean', 'zhiyongzhang/' + os.path.relpath(
            full_file_path, config.project_dir), ExtraArgs={'ACL': 'public-read'})
        print(file)
