import csv
import json
import shutil
import os
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from PIL import ImageTk, Image

import util
import config

# TODO: Set the path

results_path = config.loop_dir+'results1.csv'

###########################################################################

color_dict = {'duplicate': 'grey', 'background': 'white',
              'fish': 'red', 'starfish': 'orange'}

with open(results_path) as results_file:

    answers = csv.reader(results_file)
    headers = next(answers)
    answers = list(answers)

current_index = 0

completed = 0

for i in range(len(answers)):
    if len(answers[i]) == 31:
        current_index = i
        break

    completed += 1

window = tk.Tk()

canvas = tk.Canvas(window)

canvas.grid(row=0, columnspan=4)

progressbar_style = ttk.Style()
progressbar_style.theme_use('default')
progressbar_style.configure("TProgressbar", thickness=30, background='green')

progressbar = ttk.Progressbar(
    mode='determinate', style='TProgressbar', value=completed/len(answers)*100)

progressbar.grid(row=1, columnspan=4)

orig_highlightbackground = '#d9d9d9'


def get_image_file(path, window_width, window_height):

    image = Image.open(path)
    image = image.resize((window_width, window_height), Image.ANTIALIAS)

    return ImageTk.PhotoImage(image)


def read_and_display():

    answer = answers[current_index]

    approve_button.config(highlightbackground=orig_highlightbackground)
    reject_button.config(highlightbackground=orig_highlightbackground)

    if len(answer) == 33:
        next_button.config(state='normal')
        if answer[31] == 'X':
            approve_button.config(highlightbackground='green')
        else:
            reject_button.config(highlightbackground='red')
    else:
        next_button.config(state='disabled')

    if current_index > 0:
        previous_button.config(state='normal')
    else:
        previous_button.config(state='disabled')

    image_name = answer[27]
    annotations = json.loads(answer[28])
    height = float(answer[29])
    width = float(answer[30])

    window_height = 1800
    ratio = window_height/height

    window_width = int(width*ratio)

    unannotated_annotation = util.read_annotation_file(
        config.unannotated_annotation_dir+image_name+'.xml')

    canvas.delete('all')

    canvas.config(width=window_width, height=window_height)

    progressbar.config(length=window_width)

    canvas.ref_image = get_image_file(
        config.predict_image_dir+image_name+'.jpg', window_width, window_height)

    image_on_canvas = canvas.create_image(
        0, 0, anchor='nw', image=canvas.ref_image)

    for annotation in annotations[:-2]:

        label = annotation['label']

        xmin = annotation['xmin']
        ymin = annotation['ymin']
        xmax = annotation['xmax']
        ymax = annotation['ymax']

        canvas.create_rectangle(xmin*ratio, ymin*ratio,
                                xmax*ratio, ymax*ratio, width=6, outline=color_dict[label])

    for annotation, unannotated_gt_bbox in zip(annotations[-len(unannotated_annotation['bboxes']):], unannotated_annotation['bboxes']):

        unannotated_bbox = util.bbox(annotation['label'], annotation['xmin'],
                                     annotation['ymin'], annotation['xmax'], annotation['ymax'])

        canvas.create_rectangle(unannotated_bbox.xmin*ratio, unannotated_bbox.ymin*ratio,
                                unannotated_bbox.xmax*ratio, unannotated_bbox.ymax*ratio, width=6, outline=color_dict[label])

        canvas.create_rectangle(unannotated_gt_bbox.xmin*ratio, unannotated_gt_bbox.ymin*ratio,
                                unannotated_gt_bbox.xmax*ratio, unannotated_gt_bbox.ymax*ratio, width=6, outline='green', fill="white", stipple="gray50")

        canvas.create_text((unannotated_gt_bbox.xmin+unannotated_gt_bbox.xmax)/2*ratio, (unannotated_gt_bbox.ymin+unannotated_gt_bbox.ymax)/2*ratio,
                           fill='green', font=("Times New Roman", 18, "bold"),
                           text=format(util.get_bbox_iou(
                               unannotated_bbox, unannotated_gt_bbox), '.2f'))


def on_closing():
    if messagebox.askokcancel("Save", "Do you want to save?"):

        with open(results_path, "w") as results_file:
            writer = csv.writer(results_file, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(headers)
            writer.writerows(answers)

    if messagebox.askokcancel("Quit", "Quit?"):
        window.destroy()


def next():

    global current_index

    current_index += 1

    if current_index == len(answers):
        on_closing()
    else:
        read_and_display()


def previous():

    global current_index

    current_index -= 1

    read_and_display()


def check_new_answer():
    global completed

    answer = answers[current_index]
    if len(answer) == 31:
        answer.extend([None]*(33-len(answer)))
        completed += 1
        progressbar['value'] = completed/len(answers)*100

    return answer


def approve():
    answer = check_new_answer()
    answer[31] = 'X'
    answer[32] = ''

    next()


def reject():
    answer = check_new_answer()
    answer[31] = ''
    answer[32] = 'Not well labeled'

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
