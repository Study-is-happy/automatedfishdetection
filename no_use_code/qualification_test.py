import boto3
import config

qualification_question = open('qualification_question.xml').read()
qualification_answer = open('qualification_answer.xml').read()

mturk = boto3.client('mturk',
                     aws_access_key_id='AKIAIRD5JIXH2T5ERTGA',
                     aws_secret_access_key='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',
                     region_name='us-east-1',
                     endpoint_url='https://mturk-requester-sandbox.us-east-1.amazonaws.com')

qualification_type = mturk.create_qualification_type(
    Name='Object detection tutorial',
    Description='Object detection tutorial',
    QualificationTypeStatus='Active',
    RetryDelayInSeconds=0,
    Test=qualification_question,
    AnswerKey=qualification_answer,
    TestDurationInSeconds=600)
