import csv
import json
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

color_dict = {'background': 'white',
              'fish': 'red', 'starfish': 'orange'}

with open(results_path) as results_file:

    answers = csv.reader(results_file)
    header = next(answers)
    answers = list(answers)

current_index = 0

window = tk.Tk()

canvas = tk.Canvas(window)

canvas.grid(row=0, columnspan=6)

progressbar_style = ttk.Style()
progressbar_style.theme_use('default')
progressbar_style.configure("TProgressbar", thickness=30, background='green')

progressbar = ttk.Progressbar(
    mode='determinate', style='TProgressbar')

progressbar.grid(row=2, columnspan=6)

orig_background = '#d9d9d9'


def get_image_file(path, window_width, window_height):

    image = Image.open(path)
    image = image.resize((window_width, window_height), Image.ANTIALIAS)

    return ImageTk.PhotoImage(image)


def read_and_display():

    answer = answers[current_index]

    approve_button.config(background=orig_background)
    reject_button.config(background=orig_background)

    if answer[-2] == 'X':
        approve_button.config(background='green')
    elif answer[-1] != '':
        reject_button.config(background='red')

    if current_index < len(answers)-1:
        next_button.config(state='normal')
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

    window.title(image_name)

    window_height = 1800

    ratio = window_height/height

    window_width = int(width*ratio)

    predict_annotation = util.read_annotation_file(
        config.predict_annotation_dir+image_name+'.xml')

    gt_annotation = util.read_annotation_file(
        config.update_gt_annotation_dir+image_name+'.xml')

    exist_annotation = util.read_annotation_file(
        config.predict_exist_annotation_dir+image_name+'.xml')

    canvas.delete('all')

    canvas.config(width=window_width, height=window_height)

    canvas.ref_image = get_image_file(
        config.predict_image_dir+image_name+'.jpg', window_width, window_height)

    image_on_canvas = canvas.create_image(
        0, 0, anchor='nw', image=canvas.ref_image)

    progressbar.config(length=window_width)
    progressbar['value'] = (current_index+1)/len(answers)*100

    if exist_annotation is not None:

        for exist_bbox in exist_annotation['bboxes']:

            canvas.create_rectangle(exist_bbox.xmin*ratio, exist_bbox.ymin*ratio,
                                    exist_bbox.xmax*ratio, exist_bbox.ymax*ratio, width=6, outline='green', dash=(2, 4))

    for annotation in annotations:

        label = annotation['label']

        xmin = annotation['xmin']
        ymin = annotation['ymin']
        xmax = annotation['xmax']
        ymax = annotation['ymax']

        predict_bbox = predict_annotation['bboxes'][annotation['index']]

        canvas.create_rectangle(xmin*ratio, ymin*ratio,
                                xmax*ratio, ymax*ratio, width=6, outline=color_dict[label])

        # canvas.create_text((xmin+xmax)/2*ratio, (ymin+ymax)/2*ratio,
        #                    fill='red', font=(
        #                        "Times New Roman", 18, "bold"),
        #                    text='timer: '+format(annotation['timer']/10, '.1f')+' conf: ' + format(predict_bbox.confidence, '.1f'))

    for annotation, gt_bbox in zip(annotations[-len(gt_annotation['bboxes']):], gt_annotation['bboxes']):

        adjust_bbox = util.bbox(annotation['label'], annotation['xmin'],
                                annotation['ymin'], annotation['xmax'], annotation['ymax'])

        canvas.create_rectangle(gt_bbox.xmin*ratio, gt_bbox.ymin*ratio,
                                gt_bbox.xmax*ratio, gt_bbox.ymax*ratio, width=6, outline='green', fill="white", stipple="gray50")

        iou = util.get_bbox_iou(adjust_bbox, gt_bbox)

        canvas.create_text((gt_bbox.xmin+gt_bbox.xmax)/2*ratio, (gt_bbox.ymin+gt_bbox.ymax)/2*ratio,
                           fill='green', font=("Times New Roman", 18, "bold"),
                           text=format(iou, '.2f'))


def on_closing():
    if messagebox.askokcancel("Save", "Do you want to save?"):

        with open(results_path, "w") as results_file:
            writer = csv.writer(results_file, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(header)
            writer.writerows(answers)

    if messagebox.askokcancel("Quit", "Quit?"):
        window.destroy()


def next():

    global current_index

    if current_index == len(answers)-1:
        on_closing()
    else:
        current_index += 1
        read_and_display()


def previous():

    global current_index

    current_index -= 1

    read_and_display()


def approve():
    answer = answers[current_index]
    answer[-2] = 'X'
    answer[-1] = ''

    next()


def reject():
    answer = answers[current_index]
    answer[-2] = ''
    answer[-1] = 'Not well labeled'

    next()


previous_button = tk.Button(window, text='<- Previous',
                            command=previous, width=15, height=4, borderwidth=10, activebackground="silver")

previous_button.grid(row=3, column=0, sticky='W')

approve_button = tk.Button(window, text='Approve',
                           command=approve, width=15, height=4, borderwidth=10, activebackground="green")

approve_button.grid(row=3, column=1, sticky='E')

reject_button = tk.Button(window, text='Reject',
                          command=reject, width=15, height=4, borderwidth=10, activebackground="red")

reject_button.grid(row=3, column=2, sticky='W')

next_button = tk.Button(window, text='Next ->',
                        command=next, width=15, height=4, borderwidth=10, activebackground="silver")

next_button.grid(row=3, column=3, sticky='E')

window.protocol("WM_DELETE_WINDOW", on_closing)

read_and_display()

window.mainloop()
