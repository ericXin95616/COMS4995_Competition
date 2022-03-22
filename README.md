# COMS4995 Competition
## Problem
[Object Detection in the Hazy Condition](http://cvpr2022.ug2challenge.org/)

[Codalab link](https://codalab.lisn.upsaclay.fr/competitions/1235#learn_the_details)

## Image Dehazing
Object detection in the haze image seems to be a very difficult
task to complete. Therefore, we decide to first preprocess images
to remove haze. We have tried several haze removal techniques.

1. [FFA-Net](https://github.com/zhilin007/FFA-Net)
2. [BPPNet](https://github.com/ayu-22/BPPNet-Back-Projected-Pyramid-Network)
3. [AOD-Net](https://github.com/Boyiliee/AOD-Net)
4. [Dark Channel Prior Image Dehazing](https://github.com/He-Zhang/image_dehaze)
5. [Our own naive VAE](https://github.com/ericXin95616/COMS4995_Competition/blob/main/model/dehazing_VAE.py)
6. [NTIRE](https://github.com/liuh127/NTIRE-2021-Dehazing-Two-branch)

We have used their pretrained weights and fine tuned these models using
our dataset.

Overall, we think AOD-Net and Dark Channel Prior Image Dehazing
provides best results. These are two techniques we used to preprocessing
hazed images in dry run.

## Object Detection
We primarily used Tensorflow's object detection API to perform our object
detection task. We have experimented three different models for object detection.

[TensorFlow 2 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)

1. CenterNet HourGlass104 512x512
2. EfficientDet D4 1024x1024
3. Faster R-CNN ResNet101 V1 640x640

We first tested the performance of these models using the pretrained models and 
fine tuned these models using our training data.

We trained our models on Google Colab and stored all the files in this [shared drive](https://drive.google.com/drive/folders/1NQ6kbfawC4kjJNbWUW0uuMYnJQJg17UW?usp=sharing).

[tf_object_detection.ipynb](https://github.com/ericXin95616/COMS4995_Competition/blob/main/model/tf_object_detection.ipynb)
is the jupyter notebook we used to train our object detection models and produce the final results.

[CVPR-UG2+StarterNotebook.ipynb](https://github.com/ericXin95616/COMS4995_Competition/blob/main/model/CVPR-UG2%2BStarterNotebook.ipynb) is
the jupyter notebook we used to train NTIRE dehazing model.


