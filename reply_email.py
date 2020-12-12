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
    'FROM "<mturk-noreply@amazon.com>" ' +
    'UNANSWERED ' +
    'SINCE "12-Dec-2020"' +
    ')')

receive_email_id_list = receive_email_id_list[0].decode(
    'utf-8').split()
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
                    receive_email = receive_email_list[index]
                    email_result_list.append({'hit_id': hit_id,
                                              'assignment_id': result[14],
                                              'annotations': json.loads(result[-8]),
                                              'email_id': receive_email_id_list[index],
                                              'email_message_id': receive_email['Message-ID'],
                                              'email_subject': receive_email['Subject'],
                                              'email_from': receive_email['Reply-To'],
                                              'approve': None,
                                              'email_text': '',
                                              'reject_option': '',
                                              'confirm': False})
                    found = True
                    break
        if found:
            break

if len(email_result_list) == 0:
    imap.logout()
    sys.exit()

canvas_size = 800
email_size = 40
current_index = 0
window = tk.Tk()
canvas = tk.Canvas(window, width=canvas_size, height=canvas_size)
canvas.grid(rowspan=4, columnspan=4)
reply_email_label = tk.Label(window, text='Reply email:')
reply_email_label.grid(row=0, column=4)
progressbar_style = tkinter.ttk.Style()
progressbar_style.theme_use('default')
progressbar_style.configure('TProgressbar', thickness=30, background='green')
progressbar = tkinter.ttk.Progressbar(
    mode='determinate', style='TProgressbar', value=current_index/len(email_result_list)*100, length=canvas_size)
progressbar.grid(row=4, columnspan=4)
name_var = tk.StringVar()
orig_highlightbackground = '#d9d9d9'

input_text = tk.Text(window, width=email_size)
input_text.grid(row=1, column=4)

option_var = tk.StringVar(window)


def update_input_text(email_result):

    input_text.config(state='normal')
    input_text.delete(1.0, tk.END)
    input_text.insert(
        tk.INSERT, email_result['email_text'].format(email_result['reject_option']))
    input_text.config(state='disabled')


def update_input_text_options(email_result):

    update_input_text(email_result)

    option_menu['menu'].delete(0, 'end')

    option_var.set('')
    options = []

    if email_result['approve'] == False:
        option_var.set('Choose reject reason:')
        options = [
            'The bounding box is not fitting tightly.',
            'The bounding box is too small.'
        ]

    for option in options:
        option_menu['menu'].add_command(
            label=option, command=tk._setit(option_var, option, choose_option))


def choose_option(option):
    email_result = email_result_list[current_index]
    email_result['reject_option'] = option
    update_input_text(email_result)


option_menu = tk.OptionMenu(
    window, option_var, [])
option_menu.grid(row=2, column=4)


def read_and_display():

    canvas.delete('all')

    email_result = email_result_list[current_index]

    window.title(email_result['email_from'])

    approve_button.config(highlightbackground=orig_highlightbackground)
    reject_button.config(highlightbackground=orig_highlightbackground)

    if not email_result['confirm']:
        next_button.config(state='disabled')
    else:
        next_button.config(state='normal')
        if email_result['approve']:
            approve_button.config(highlightbackground='green')
        else:
            reject_button.config(highlightbackground='red')

    if current_index > 0:
        previous_button.config(state='normal')
    else:
        previous_button.config(state='disabled')

    annotations = email_result['annotations']
    gt_annotation = annotations[config.gt_indexes[0]]
    print(gt_annotation['image_id'])

    image_width = gt_annotation['width']
    image_height = gt_annotation['height']

    bbox = np.copy(gt_annotation['bbox'])
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

    cv2.imwrite('/tmp/' + email_result['assignment_id'] + '.jpg', image)

    image = cv2.resize(image, (canvas_size, canvas_size))

    canvas.ref_image = PIL.ImageTk.PhotoImage(
        image=PIL.Image.fromarray(np.flip(image, axis=-1)))

    canvas.create_image(0, 0, anchor='nw', image=canvas.ref_image)

    update_input_text_options(email_result)


def reply():
    s3_client = boto3.client('mturk',
                             aws_access_key_id=config.aws_access_key_id,
                             aws_secret_access_key=config.aws_secret_access_key,
                             region_name='us-east-1',
                             endpoint_url='https://mturk-requester.us-east-1.amazonaws.com')

    smtp = smtplib.SMTP_SSL('smtp.gmail.com')
    smtp.login(config.email_address, config.email_password)

    imap.select()

    for email_result in email_result_list:

        if email_result['confirm']:

            assignment_id = email_result['assignment_id']

            if email_result['approve']:
                assignment_status = s3_client.get_assignment(
                    AssignmentId=assignment_id)['Assignment']['AssignmentStatus']
                if (assignment_status != 'Approved'):
                    s3_client.approve_assignment(
                        AssignmentId=assignment_id,
                        RequesterFeedback='Approved, thank you for your contribution!',
                        OverrideRejection=True
                    )

            else:
                s3_client.approve_assignment(
                    AssignmentId=assignment_id,
                    RequesterFeedback=email_result['reject_option'],
                    OverrideRejection=False
                )

            imap.store(email_result['email_id'], '+FLAGS', '\\Answered \\SEEN')

            reply_email = email.mime.multipart.MIMEMultipart()

            reply_email['References'] = reply_email['In-Reply-To'] = email_result['email_message_id']
            reply_email['Subject'] = 'Re: ' + email_result['email_subject']
            reply_email['From'] = config.email_name
            reply_email['To'] = email_result['email_from']

            reply_email_text = 'Hello '+email_result['email_from'].split()[0]+',\n\n'+email_result['email_text'].format(
                email_result['reject_option'])+'\n\nBest,\n'+config.email_name

            reply_email.attach(email.mime.text.MIMEText(reply_email_text))

            with open('/tmp/'+assignment_id+'.jpg', 'rb') as image_file:

                mime_image = email.mime.image.MIMEImage(image_file.read())
                reply_email.attach(mime_image)

    imap.close()
    imap.logout()
    smtp.quit()


def on_closing():
    if tkinter.messagebox.askokcancel('Reply', 'Do you want to reply?'):
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
    reject_button.config(highlightbackground=orig_highlightbackground)
    approve_button.config(highlightbackground='green')
    email_result = email_result_list[current_index]
    email_result['approve'] = True
    email_result['email_text'] = '\tThank you for working on our tasks! The attached image was flagged by our program for being outside range of acceptable labelling. However, after reviewing this image, it appears your labelling was rejected in error. We have updated the status of the batch of images which were rejected because of this to “accepted.” We appreciate you taking the time to work on our tasks, and thank you for your understanding in this matter, as we are still fine-tuning the rejection parameters.'
    email_result['reject_option'] = ''
    update_input_text_options(email_result)


def reject():
    approve_button.config(highlightbackground=orig_highlightbackground)
    reject_button.config(highlightbackground='red')
    email_result = email_result_list[current_index]
    email_result['approve'] = False
    email_result['email_text'] = '\tThank you for working on our tasks! The attached image was flagged by our program for being outside range of acceptable labelling. After reviewing this image, it appears your labelling was rejected for the following reason:' + \
        '\n\n{}\n\n' + \
        '\tUnfortunately, we believe this labelling does not meet our requirements. We apologize for our strict rejection parameters, our use case requires very precise labelling. We appreciate you taking the time to work on our tasks and hope you will consider working on more in the future, as we are continuing to fine-tune our rejection parameters.'
    email_result['reject_option'] = ''
    update_input_text_options(email_result)


def confirm():
    email_result = email_result_list[current_index]
    if email_result['approve'] is None:
        tkinter.messagebox.showwarning(message='Approve or reject!')
        return
    if email_result['approve'] == False and email_result['reject_option'] == '':
        tkinter.messagebox.showwarning(message='Choose reject reason!')
        return
    email_result['confirm'] = True
    next()


previous_button = tk.Button(window, text='<- Previous',
                            command=previous, width=15, height=4, borderwidth=10)

previous_button.grid(row=5, column=0, sticky='W')

approve_button = tk.Button(window, text='Approve',
                           command=approve, width=15, height=4, borderwidth=10, highlightthickness=10, activebackground='green')

approve_button.grid(row=5, column=1, sticky='E')

reject_button = tk.Button(window, text='Reject',
                          command=reject, width=15, height=4, borderwidth=10, highlightthickness=10, activebackground='red')

reject_button.grid(row=5, column=2, sticky='W')

next_button = tk.Button(window, text='Next ->', command=next,
                        width=15, height=4, borderwidth=10)

next_button.grid(row=5, column=3, sticky='E')

confirm_button = tk.Button(window, text='Confirm',
                           command=confirm, width=15, height=4, borderwidth=10, highlightthickness=10)

confirm_button.grid(row=3, column=4)

window.protocol('WM_DELETE_WINDOW', on_closing)

read_and_display()

window.mainloop()
