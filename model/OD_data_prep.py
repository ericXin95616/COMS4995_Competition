# prepare data by transforming the dataset to TFRecord
# refer link: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md

from utils import *

import tensorflow._api.v2.compat.v1 as tf

from object_detection.utils import dataset_util
import glob


flags = tf.app.flags
# flags.DEFINE_string('output_path', '../workspace/data/test.record', 'Path to output TFRecord')
# flags.DEFINE_string('output_path', '../workspace/data/train.record', 'Path to output TFRecord')
flags.DEFINE_string('output_path', '../workspace/data/clean_haze_train.record', 'Path to output TFRecord')
#flags.DEFINE_string('output_path', '../workspace/data/clean_haze_test.record', 'Path to output TFRecord')
FLAGS = flags.FLAGS

BPP_root = '../BPP_train'
TRAIN_root = '../train'

path_of_train_hazy_images = os.path.join(BPP_root, 'haze_train')
path_of_train_clean_images = os.path.join(BPP_root, 'dehaze_train')

path_of_test_hazy_images = os.path.join(BPP_root, 'haze_test')
path_of_test_clean_images = os.path.join(BPP_root, 'dehaze_test')

path_of_clean_images = os.path.join(TRAIN_root, 'clean_images')

images_paths_haze_train = glob.glob(path_of_train_hazy_images + '/*.jpg')
images_paths_haze_test = glob.glob(path_of_test_hazy_images + '/*.jpg')

images_paths_dehaze_train = glob.glob(path_of_train_clean_images + '/*.jpg')
images_paths_dehaze_test = glob.glob(path_of_test_clean_images + '/*.jpg')

images_paths_clean_train = glob.glob(path_of_clean_images + '/*.jpg')
print(len(images_paths_clean_train))

tmp = int(0.8 * len(images_paths_clean_train))
clean_haze_trainset = images_paths_haze_train + images_paths_clean_train[0: tmp]
print(clean_haze_trainset)
clean_haze_valset = images_paths_haze_test + images_paths_clean_train[tmp:]
print(clean_haze_valset)

def create_tf_example(filename):
    # read haze images
    with open(filename, 'rb') as f:
        encoded_image_data = f.read()
    label = get_images_bb_map([filename], 'labels')
    height = 750 # Image height
    width = 1845 # Image width
    filename = filename # Filename of the image. Empty if image is not from file
     # Encoded image bytes
    image_format = b'jpg' # b'jpeg' or b'png'

    xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = []  # List of normalized right x coordinates in bounding box
    # (1 per box)
    ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = []  # List of normalized bottom y coordinates in bounding box
    # (1 per box)
    classes_text = []  # List of string class name of bounding box (1 per box)
    classes = []  # List of integer class id of bounding box (1 per box)

    for coord in label[filename]:
        xmin, ymin = int(coord[0]), int(coord[1])
        xmax, ymax = int(coord[2]), int(coord[3])
        xmins.append(xmin / width)
        xmaxs.append(xmax / width)
        ymins.append(ymin / height)
        ymaxs.append(ymax / height)
        classes_text.append(b'vehicle')
        classes.append(1)

    print(xmins)
    print(xmaxs)
    assert 750 == height
    assert 1845 == width
    filename = '{}.jpg'.format(Path(filename).stem)
    tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename.encode()),
      'image/source_id': dataset_util.bytes_feature(filename.encode()),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    #for im in images_paths_haze_train:
    #for im in images_paths_haze_test:
    #for im in images_paths_dehaze_train:
    for im in clean_haze_trainset:
        tf_example = create_tf_example(im)
        writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()

