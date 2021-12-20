from xml.etree import ElementTree
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

import config

# TODO: Set the dir

dataset_dir = config.project_dir + 'xml/20181016/'

###########################################################################

annotations_dir = dataset_dir + 'annotations/'

images_dir = dataset_dir + 'images/'

color_dict = {'fish': 'red',
              'starfish': 'orange',
              'sponge': 'white',
              'sponges': 'white',
              'rockfish': 'red',
              'corals': 'orange',
              'invertebrates': 'gray',
              'roundfish': 'gray',
              'flatfish': 'gray',
              'skates': 'gray',
              'unknown': 'gray',
              'background': 'gray'
              }

for annotation_file in os.listdir(annotations_dir):

    annotation_node = ElementTree.parse(annotations_dir + annotation_file)

    image_file_name = annotation_node.find('filename').text

    image = mpimg.imread(images_dir + image_file_name)

    fig = plt.figure(figsize=(20, 20))

    plt.title(image_file_name)

    plt.imshow(image)

    current_axis = plt.gca()

    object_nodes = annotation_node.findall('object')

    for object_node in object_nodes:

        category = object_node.find('name').text.lower()

        color = color_dict[category]

        bndbox_node = object_node.find('bndbox')

        bbox = [int(bndbox_node.find('xmin').text) - 1,
                int(bndbox_node.find('ymin').text) - 1,
                int(bndbox_node.find('xmax').text),
                int(bndbox_node.find('ymax').text)]

        current_axis.add_patch(plt.Rectangle(
            (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], color=color, fill=False, linewidth=3))

        plt.text(bbox[0], bbox[1] - 3,
                 category, color='white', size=10, bbox={'facecolor': color, 'alpha': 0.5, 'pad': 3})

    plt.show()
