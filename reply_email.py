import imaplib
import os
import csv
import json
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import ImageTk, Image
import threading
import boto3
import smtplib
import email.mime.text
import email.mime.multipart
import email.mime.image


import config
import util

with imaplib.IMAP4_SSL('imap.gmail.com') as imap:
    imap.login(config.email_address, config.email_password)

    imap.select('inbox')
    _, email_id_list = imap.search(
        None, '(' +
        'SUBJECT "[Amazon Mechanical Turk] Regarding Amazon Mechanical Turk HIT" ' +
        'FROM <mturk-noreply@amazon.com> ' +
        'UNANSWERED ' +
        'SINCE "07-Dec-2020"' +
        ')')

    hit_id_list = []

    for email_id in email_id_list[0].decode('utf-8').split():
        # BODY[TEXT]
        _, body = imap.fetch(email_id, '(BODY.PEEK[TEXT])')

        body = body[0][1].decode('utf-8')
        for line in body.splitlines():
            if line.startswith('HIT ID:'):
                hit_id_list.append(line.split()[-1])
                break

    imap.close()

email_results = []

results_approve_dir = config.project_dir + 'results_approve/'

for results_name in os.listdir(results_approve_dir):
    with open(results_approve_dir + results_name) as results_approve_file:

        results = csv.reader(results_approve_file)

        next(results)

        for result in results:
            if result[0] in hit_id_list:
                result[-5] = ''
                result[-6] = ''
                email_results.append(result)

window_size = 800
current_index = 0
window = tk.Tk()
canvas = tk.Canvas(window, width=window_size, height=window_size)
canvas.grid(row=0, columnspan=4)
progressbar_style = ttk.Style()
progressbar_style.theme_use('default')
progressbar_style.configure("TProgressbar", thickness=30, background='green')
progressbar = ttk.Progressbar(
    mode='determinate', style='TProgressbar', value=current_index/len(hit_id_list)*100, length=window_size)
progressbar.grid(row=1, columnspan=4)
orig_highlightbackground = '#d9d9d9'


def read_and_display():

    canvas.delete('all')

    result = email_results[current_index]

    approve_button.config(highlightbackground=orig_highlightbackground)
    reject_button.config(highlightbackground=orig_highlightbackground)

    if result[-5] == '' and result[-6] == '':
        next_button.config(state='disabled')
    else:
        next_button.config(state='normal')
        if result[-6] == 'x':
            approve_button.config(highlightbackground='green')
        else:
            reject_button.config(highlightbackground='red')

    if current_index > 0:
        previous_button.config(state='normal')
    else:
        previous_button.config(state='disabled')

    annotations = json.loads(result[-8])
    gt_annotation = annotations[config.gt_indexes[0]]

    image_width = gt_annotation['width']
    image_height = gt_annotation['height']

    bbox = gt_annotation['bbox']
    util.rel_to_abs(bbox, image_width, image_height)

    image = cv2.imread(config.project_dir + 'update/images/' +
                       gt_annotation['image_id'] + '.jpg', -1)

    image = cv2.rectangle(
        image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)

    half_image_size = max(bbox[2]-bbox[0], bbox[3]-bbox[1])*1.8/2

    center_x = (bbox[0]+bbox[2])/2
    center_y = (bbox[1]+bbox[3])/2

    offset_left = int(max(0, center_x-half_image_size))
    offset_right = int(max(0, center_x+half_image_size))

    offset_top = int(max(0, center_y-half_image_size))
    offset_bottom = int(max(0, center_y+half_image_size))

    image = image[offset_top:offset_bottom, offset_left:offset_right, :]

    cv2.imwrite('/tmp/' + result[14] + '.jpg', image)

    image = cv2.resize(image, (window_size, window_size))

    canvas.ref_image = ImageTk.PhotoImage(
        image=Image.fromarray(np.flip(image, axis=-1)))

    canvas.create_image(0, 0, anchor='nw', image=canvas.ref_image)

    ratio = window_size/(half_image_size*2)


def reply():
    s3_client = boto3.client('mturk',
                             aws_access_key_id=config.aws_access_key_id,
                             aws_secret_access_key=config.aws_secret_access_key,
                             region_name='us-east-1',
                             endpoint_url='https://mturk-requester.us-east-1.amazonaws.com')

    smtp = smtplib.SMTP('smtp.gmail.com')
    smtp.starttls()
    smtp.login(config.email_address, config.email_password)

    for result in email_results:

        assignment_id = result[14]

        if result[-6] == 'x':
            email_text = 'assignment id: ' + result[14] + ': approve'
    #         s3_client.approve_assignment(
    #             AssignmentId=result[14],
    #             OverrideRejection=True
    #         )
        elif result[-5] != '':
            email_text = 'assignment id: ' + result[14] + ': reject'

        else:
            continue

        email_message = email.mime.multipart.MIMEMultipart()

        email_message['From'] = 'Hanu'
        email_message['To'] = 'zhiyong'
        email_message['Subject'] = 'auto reply test'

        email_message.attach(email.mime.text.MIMEText(email_text))

        image_file_name = assignment_id+'.jpg'

        with open('/tmp/'+image_file_name, 'rb') as image_file:

            mime_image = email.mime.image.MIMEImage(image_file.read())
            email_message.attach(mime_image)

        # smtp.sendmail(config.email_address,
        #               'zhang.zhiyo@northeastern.edu', str(email_message))

    smtp.quit()


def on_closing():
    if messagebox.askokcancel("Reply", "Do you want to reply?"):
        threading.Thread(target=reply).start()

    if messagebox.askokcancel("Quit", "Quit?"):
        window.destroy()


def next():

    global current_index

    current_index += 1
    progressbar['value'] = current_index/len(hit_id_list)*100

    if current_index == len(hit_id_list):
        on_closing()
    else:
        read_and_display()


def previous():

    global current_index

    current_index -= 1

    read_and_display()


def approve():
    email_results[current_index][-6] = 'x'
    email_results[current_index][-5] = ''

    next()


def reject():
    email_results[current_index][-6] = ''
    email_results[current_index][-5] = 'We checked again, the bounding box is bad'

    next()


previous_button = tk.Button(window, text='<- Previous',
                            command=previous, width=15, height=4, borderwidth=10)

previous_button.grid(row=2, column=0, sticky='W')

approve_button = tk.Button(window, text='Approve',
                           command=approve, width=15, height=4, borderwidth=10, highlightthickness=10, activebackground="green")

approve_button.grid(row=2, column=1, sticky='E')

reject_button = tk.Button(window, text='Reject',
                          command=reject, width=15, height=4, borderwidth=10, highlightthickness=10, activebackground="red")

reject_button.grid(row=2, column=2, sticky='W')

next_button = tk.Button(window, text='Next ->', command=next,
                        width=15, height=4, borderwidth=10)

next_button.grid(row=2, column=3, sticky='E')

window.protocol("WM_DELETE_WINDOW", on_closing)

read_and_display()

window.mainloop()
