import imaplib
import os
import csv
import json
import cv2
import numpy as np
import tkinter as tk
import tkinter
import tkinter.ttk
import tkinter.messagebox
import PIL.ImageTk
import threading
import boto3
import smtplib
import email.mime.text
import email.mime.multipart
import email.mime.image
import sys

import config
import util

imap = imaplib.IMAP4_SSL('imap.gmail.com')
imap.login(config.email_address, config.email_password)

imap.select(readonly=True)
_, receive_email_id_list = imap.search(
    None, '(' +
    'SUBJECT "[Amazon Mechanical Turk] Regarding Amazon Mechanical Turk HIT" ' +
    # 'FROM "<mturk-noreply@amazon.com>" ' +
    'FROM "Zhiyong Zhang" ' +
    # 'UNANSWERED ' +
    'SINCE "07-Dec-2020"' +
    ')')

receive_email_id_list = receive_email_id_list[0].decode('utf-8').split()
hit_id_list = []
receive_email_list = []

for receive_email_id in receive_email_id_list:
    _, receive_email = imap.fetch(receive_email_id, '(RFC822)')

    receive_email = email.message_from_bytes(receive_email[0][1])
    receive_email_list.append(receive_email)
    hit_id_list.append(receive_email['Subject'].split()[-1])

imap.close()

email_result_list = []

results_approve_dir = config.project_dir + 'results_approve/'

for index, hit_id in enumerate(hit_id_list):
    found = False
    for results_name in os.listdir(results_approve_dir):
        with open(results_approve_dir + results_name) as results_approve_file:

            results = csv.reader(results_approve_file)

            next(results)

            for result in results:
                if result[0] == hit_id:
                    result[-5] = ''
                    result[-6] = ''
                    email_result_list.append(result)
                    found = True
                    break
        if found:
            break
    else:
        del receive_email_id_list[index]
        del receive_email_list[index]

if len(email_result_list) == 0:
    imap.logout()
    sys.exit()

window_size = 800
current_index = 0
window = tk.Tk()
canvas = tk.Canvas(window, width=window_size, height=window_size)
canvas.grid(row=0, columnspan=4)
progressbar_style = tkinter.ttk.Style()
progressbar_style.theme_use('default')
progressbar_style.configure("TProgressbar", thickness=30, background='green')
progressbar = tkinter.ttk.Progressbar(
    mode='determinate', style='TProgressbar', value=current_index/len(email_result_list)*100, length=window_size)
progressbar.grid(row=1, columnspan=4)
orig_highlightbackground = '#d9d9d9'
# reject_options = [
#     "Jan",
#     "Feb",
#     "Mar"
# ]
# option_variable = tk.StringVar(canvas)
# option_variable.set(reject_options[0])


def read_and_display():

    canvas.delete('all')

    result = email_result_list[current_index]

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

    rint_bbox = util.get_rint(bbox)

    image = cv2.rectangle(
        image, (rint_bbox[0], rint_bbox[1]), (rint_bbox[2], rint_bbox[3]), (0, 0, 255), 2)

    half_image_size = max(bbox[2]-bbox[0], bbox[3]-bbox[1])*1.8/2

    center_x = (bbox[0]+bbox[2])/2
    center_y = (bbox[1]+bbox[3])/2

    offset_left = util.get_rint(max(0, center_x-half_image_size))
    offset_right = util.get_rint(max(0, center_x+half_image_size))

    offset_top = util.get_rint(max(0, center_y-half_image_size))
    offset_bottom = util.get_rint(max(0, center_y+half_image_size))

    image = image[offset_top:offset_bottom, offset_left:offset_right, :]

    cv2.imwrite('/tmp/' + result[14] + '.jpg', image)

    image = cv2.resize(image, (window_size, window_size))

    canvas.ref_image = PIL.ImageTk.PhotoImage(
        image=PIL.Image.fromarray(np.flip(image, axis=-1)))

    canvas.create_image(0, 0, anchor='nw', image=canvas.ref_image)

    # optio_menu = tk.OptionMenu(canvas, option_variable, *reject_options)


def reply():
    s3_client = boto3.client('mturk',
                             aws_access_key_id=config.aws_access_key_id,
                             aws_secret_access_key=config.aws_secret_access_key,
                             region_name='us-east-1',
                             endpoint_url='https://mturk-requester.us-east-1.amazonaws.com')

    smtp = smtplib.SMTP_SSL('smtp.gmail.com')
    smtp.login(config.email_address, config.email_password)

    imap.select()

    for result, receive_email_id, receive_email in zip(email_result_list, receive_email_id_list, receive_email_list):

        assignment_id = result[14]

        if result[-6] == 'x':
            reply_email_text = 'assignment id: ' + assignment_id + ': approve'
            assignment_status = s3_client.get_assignment(
                AssignmentId=assignment_id)['Assignment']['AssignmentStatus']
            if (assignment_status != 'Approved'):
                s3_client.approve_assignment(
                    AssignmentId=assignment_id,
                    RequesterFeedback='Approved, thank you for your contribution!',
                    OverrideRejection=True
                )

        elif result[-5] != '':
            reply_email_text = 'assignment id: ' + assignment_id + ': reject'
            # s3_client.approve_assignment(
            #     AssignmentId=assignment_id,
            #     RequesterFeedback='',
            #     OverrideRejection=False
            # )

        else:
            continue

        imap.store(receive_email_id, '+FLAGS', '\\Answered \\SEEN')

        reply_email = email.mime.multipart.MIMEMultipart()

        reply_email['References'] = reply_email['In-Reply-To'] = receive_email['Message-ID']
        reply_email['Subject'] = 'Re: ' + receive_email['Subject']
        reply_email['From'] = 'Field Robotics Lab'
        reply_email['To'] = receive_email['From']

        reply_email.attach(email.mime.text.MIMEText(reply_email_text))

        image_file_name = assignment_id+'.jpg'
        image_file_path = '/tmp/'+image_file_name

        with open(image_file_path, 'rb') as image_file:

            mime_image = email.mime.image.MIMEImage(image_file.read())
            reply_email.attach(mime_image)

        smtp.sendmail(config.email_address,
                      receive_email['From'], reply_email.as_bytes())

        os.remove(image_file_path)

    imap.close()
    imap.logout()
    smtp.quit()


def on_closing():
    if tkinter.messagebox.askokcancel("Reply", "Do you want to reply?"):
        threading.Thread(target=reply).start()

    else:
        imap.logout()

    window.destroy()


def next():

    global current_index

    current_index += 1
    progressbar['value'] = current_index/len(email_result_list)*100

    if current_index == len(email_result_list):
        on_closing()
    else:
        read_and_display()


def previous():

    global current_index

    current_index -= 1

    read_and_display()


def approve():
    result = email_result_list[current_index]
    result[-6] = 'x'
    result[-5] = ''

    next()


def reject():
    result = email_result_list[current_index]
    result[-6] = ''
    result[-5] = 'We checked again, the bounding box is bad'

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
