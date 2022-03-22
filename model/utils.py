"""**Import Libraries**"""

import os
from glob import glob
from pathlib import Path
from collections import defaultdict
from pprint import pprint
from random import choice, sample
import cv2
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.patches import Rectangle

# check if gpu is available
gpu = tf.config.list_physical_devices(device_type='GPU')
if 0 == len(gpu):
    print("No GPU available!")
else:
    print("Num GPUs: ", len(gpu))

"""
train

 -clean_images

 -clean_images_labels

 -dry-run

 -haze_images

 -haze_images_labels
"""

# Change this path to your path to your train folder
root = '../train'
dry_run_root = '../dry-run'

haze_images = os.path.join(root, 'haze_images')
haze_images_labels = os.path.join(root, 'haze_images_labels')

# Not used in the starter notebook. Feel free to use it
clean_images = os.path.join(root, 'clean_images')
clean_images_labels = os.path.join(root, 'clean_images_labels')
dry_run_images_dir = os.path.join(dry_run_root, 'images')
dry_run_images_labels_dir = os.path.join(dry_run_root, 'labels')
dry_run_images = glob(dry_run_images_dir + '/*.jpg')

# Check the path
assert os.path.exists(haze_images)
assert os.path.exists(haze_images_labels)

# Image with name greater or equal to 170 will be considered as validation images.
# Example 170.jpg, 173.jpg etc
val_cutoff = 170

all_haze_images = glob(haze_images + "/*.jpg")
print("all haze images")
print(all_haze_images)

train_images = [x for x in all_haze_images if int(Path(x).stem) < val_cutoff]
val_images = [x for x in all_haze_images if int(Path(x).stem) >= val_cutoff]

print('Number of traing images :', len(train_images))
print('Number of validation images :', len(val_images))

assert len(train_images) + len(val_images) == len(all_haze_images)


def get_images_bb_map(images_list, directory):
    '''
    Returns the mapping of image path with the ground truth bounding box coords.
    This method fetches the coordinates of the bounding box from the haze_image_labels
    directory.

    Parameters:
          images_list : List of images
          directory: labels directory

    Returns:
          train_images_box_map : A dictionary with Image path as the key and
          the value is the list of detected objects with each item in a list being
          a list of coordinates which identifies the bounding box of the detected object
          in the image
    '''
    images_box_map = defaultdict(list)
    #label_directory = os.path.join(root, directory)
    assert os.path.exists(directory)
    for x in images_list:
        img_without_extention = Path(x)
        image_label = os.path.join(directory, f'{img_without_extention.stem}.txt')
        with open(image_label, "r") as f:
            lines = f.read().strip().split('\n')
            for line in lines:
                line_text = line.strip().split(' ')
                assert len(line_text) == 5, 'Each line must have 5 values'  # one identifer and 4 coordinates
                coords = line_text[1:]
                images_box_map[x].append(coords)
    return images_box_map


# mapping images to the detected objects
train_images_box_map = get_images_bb_map(train_images, directory=haze_images_labels)
val_images_box_map = get_images_bb_map(val_images, directory=haze_images_labels)
dry_images_box_map = get_images_bb_map(dry_run_images, dry_run_images_labels_dir)

assert len(train_images_box_map) == len(train_images)
assert len(val_images_box_map) == len(val_images)

# print a mapping example
example_train_image = list(train_images_box_map.keys())[0]
print('Key :', example_train_image)
print('Value')
pprint(train_images_box_map[example_train_image])

"""
Here we are defining an example data generator. Here our data generator will generate a batch of examples which will be
used by your model in training. It will generate a list of 'batch_size' images. Each item being a tuple of image numpy 
array and corresponding detected object's list of coordinates.
"""


def read_img(path):
    '''
    Returns the image as a numpy array

    Parameters:
        path : Path of the image

    Returns:
        img : numpy representation of the image
    '''
    img = image.load_img(path)
    img = np.array(img).astype(np.float64)
    return img


def gen(images_to_bb_map, batch_size=16):
    '''
    Returns the generator of list of tuples of image numpy array with
    the bounding box coordinates

    Parameters:
            images_to_bb_map : A dictionary with key as the image path and
            value as the list of bounding box coordinates.
    Returns:
            (image, [bounding box coordinates]) : a list of tuples. The first
            item in the tuple is a numpy array of the image and the second item
            is a list of bounding box coordinates
    '''

    images = list(images_to_bb_map.keys())
    while True:
        batch_images = sample(images, batch_size)
        ret = []
        for image in batch_images:
            ret.append((read_img(image), images_to_bb_map[image]))
        yield ret


for t in gen(train_images_box_map):
    print(len(t))
    print('Image shape : ', t[0][0].shape)
    pprint(t[0][1])
    break

"""As you can see there are 16 images and the images have their corresponding bounding box.

**Visualizing images**
"""

image_path = os.path.join(haze_images, '035.jpg')
image = plt.imread(image_path)
# plt.imshow(image)


def visualize_bb_image(img_path, coords):
    '''
    Displays the image with the detected object boundaries marked.
    This method should be used for training images only as it tries
    to fetch the bounding box from the train data directory.

    Parameters:
            img_path : Path of the image
            coord: detect coordinates
    '''
    image = plt.imread(img_path)
    # assert img_path in train_images_box_map, 'Image path not found in bounding box mapping dictionary'
    rectangles = []
    for coord in coords:
        xmin, ymin = int(coord[0]), int(coord[1])
        xmax, ymax = int(coord[2]), int(coord[3])
        width = xmax - xmin
        height = ymax - ymin

        # Draw a rectangle with blue line borders of thickness of 2 px
        rect = Rectangle((xmin, ymin), width, height, angle=0.0, linewidth=1, edgecolor='blue', facecolor='none')
        rectangles.append(rect)

    fig, ax = plt.subplots()
    ax.imshow(image)
    for rect in rectangles:
        ax.add_patch(rect)
    #plt.show()
    path = dry_run_root + '/groundtruth/' + Path(img_path).stem + '.jpg'
    plt.savefig(path)


# The actual image has a height of 1500. The image
# is made up of two parts. The top half is the hazy image
# and the bottom half is that of a clear image.
# The actual test data will be that of a hazy image with the
# height as (750, 1845)
im_height, im_width = 750, 1845

#coords = [[0, 98, 2, 225], [0, 198, 2, 370], [0, 244, 2, 426], [0, 290, 2, 476], [0, 336, 2, 524], [0, 383, 2, 570], [0, 476, 2, 661], [0, 1744, 2, 1845], [0, 523, 2, 707], [0, 1740, 2, 1812], [0, 569, 2, 753], [0, 616, 2, 798], [0, 662, 2, 844], [0, 708, 2, 890], [0, 755, 2, 936], [0, 801, 2, 982], [0, 847, 2, 1028], [0, 893, 2, 1074], [0, 940, 2, 1120], [0, 986, 2, 1166], [0, 1032, 2, 1211]]
for key in dry_images_box_map:
    visualize_bb_image(key, dry_images_box_map[key])


def get_hazy_clean_image(img_path):
    '''
    The image comprises of hazy image and clean image
    The top half is the hazy image and the bottom half is clean image

      Parameters:
              img_path : Path of the image

      Returns:
              The hazy image as a numpy array
    '''

    img = plt.imread(img_path)
    cropped_image = img[:int(img.shape[0] / 2)]
    clean_image = img[int(img.shape[0]/2):]
    return cropped_image, clean_image


hazy_image, clean_image = get_hazy_clean_image(image_path)
print(hazy_image.shape)
print(clean_image.shape)

'''
for im in train_images:
    haze_im, clean_im = get_hazy_clean_image(im)
    cv2.imwrite('../BPP_train/haze_train/' + str(Path(im).stem) + '.jpg', haze_im)
    cv2.imwrite('../BPP_train/dehaze_train/' + str(Path(im).stem) + '.jpg', clean_im)

for im in val_images:
    haze_im, clean_im = get_hazy_clean_image(im)
    cv2.imwrite('../BPP_train/haze_test/' + str(Path(im).stem) + '.jpg', haze_im)
    cv2.imwrite('../BPP_train/dehaze_test/' + str(Path(im).stem) + '.jpg', clean_im)
'''

