"""
After preparing the TFRecord, we fine-tunes pretrained model using TFRecords.
Load the fine-tuned models, try to use it to detect vehicle in the image.
"""
"""
python model_main_tf2.py --pipeline_config_path=models/faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8/v1/pipeline.config 
--model_dir=models/faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8/v1/ --checkpoint_every_n=100 --num_workers=1 alsologtostderr
"""
import os
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob


PATH_TO_MODEL_DIR = '../workspace/models/centernet_hg104_512x512_coco17_tpu-8/v2'
PATH_TO_CFG = PATH_TO_MODEL_DIR + '/pipeline.config'
PATH_TO_CKPT = PATH_TO_MODEL_DIR + '/checkpoints'
PATH_TO_LABELS = '../workspace/data/label_map.pbtxt'

PATH_TO_DATASET_DIR = '../BPP_train'
PATH_TO_VALIDATION_DIR = PATH_TO_DATASET_DIR + '/haze_test'
PATH_TO_TRAIN_DIR = PATH_TO_DATASET_DIR + '/haze_train'
PATH_TO_CLEAN_VALIDATION_DIR = PATH_TO_DATASET_DIR + '/dehaze_test'
PATH_TO_CLEAN_TRAIN_DIR = PATH_TO_DATASET_DIR + '/dehaze_train'
PATH_TO_RESULT_LABEL_DIR = './results'
PATH_TO_DRYRUN_DIR = '../train/dry-run-1'


validation_images = glob.glob(PATH_TO_VALIDATION_DIR + '/*.jpg')
train_images = glob.glob(PATH_TO_TRAIN_DIR + '/*.jpg')
clean_validation_images = glob.glob(PATH_TO_CLEAN_VALIDATION_DIR + '/*.jpg')
clean_train_images = glob.glob(PATH_TO_CLEAN_TRAIN_DIR + '/*.jpg')
dry_run_images = glob.glob(PATH_TO_DRYRUN_DIR + '/*.jpg')
dehaze_dry_run_images = glob.glob('./AOD-Net/data/result/*.jpg')


# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
print(model_config)
detection_model = model_builder.build(model_config=model_config, is_training=False)
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-35')).expect_partial()
im_width = 1845
im_height = 750
from pathlib import Path


@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    # print(detections)

    return detections

'''
test_im_path = train_images[0]
hazy_im, _ = get_hazy_clean_image(test_im_path)
tensor_im = tf.convert_to_tensor(hazy_im)
converted_img = tf.image.convert_image_dtype(tensor_im, tf.float32)[tf.newaxis, ...]
detections = detect_fn(converted_img)
# detections = detection_model(converted_img)
result = {key: value.numpy() for key, value in detections.items()}
'''

def filter_coords(result, threshold=0.0):
    '''
      A helper method which extracts the bounding boxes and threshold confidence scores
      from the data structure returned from the pretrained model

      Parameters:
              result : Output of the pretrained tensor flow model
              threshold : Optional threshold confidence above which bounding boxes are selected
      Returns:
              The list of coordinates of the detected objects and the threshold score
    '''
    class_entities = result['detection_classes']
    detection_scores = result['detection_scores']
    bounding_boxes = result['detection_boxes']
    detection_boxes_with_thresholds = []
    for detection_class, detection_score, bounding_box in zip(class_entities, detection_scores, bounding_boxes):
        if 1 == detection_class and detection_score >= threshold:
            detection_boxes_with_thresholds.append((bounding_box, detection_score))
    return detection_boxes_with_thresholds


# f = filter_coords(result, threshold=0.5)
'''
coords = result['detection_boxes']
width = 750
height = 1845
converted_coords = []
print(coords)
for coord in coords[0]:
    print(coord)
    ymin, xmin, ymax, xmax = coord
    converted_coord = [int(xmin * width), int(ymin * height), int(xmax * width), int(ymax * height)]
    converted_coords.append(converted_coord)

print(converted_coords)
visualize_bb_image(val_images[0], converted_coords)
'''


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))


for image_path in dehaze_dry_run_images:

    print('Running inference for {}... '.format(image_path), end='')

    image_np = load_image_into_numpy_array(image_path)

    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    label_id_offset = 1
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64) + label_id_offset

    image_np_with_detections = image_np.copy()

    # visualize the detected box on the image
    viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=10,
            min_score_thresh=0.2,
            agnostic_mode=False)

    plt.figure()
    filename = '{}.jpg'.format(Path(image_path).stem)
    plt.imshow(image_np_with_detections)
    plt.savefig('results_fig/'+filename)

    # output the detected box to the file
    coords = filter_coords(detections, threshold=0.2)
    newfile = os.path.join(PATH_TO_RESULT_LABEL_DIR, Path(image_path).stem + '.txt')
    with open(newfile, "w+") as f:
        for coord in coords:
            ymin, xmin, ymax, xmax = coord[0]
            threshold = coord[1]
            format_string = f'vehicle {xmin * im_width} {ymin * im_height} {xmax * im_width} {ymax * im_height} {threshold}\n'
            f.write(format_string)
    print('Done')

