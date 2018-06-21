
# coding: utf-8

# # Add Necessary Directories to Path Variable
import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import *

home_dir="/home/paperspace/"
module_path = (os.path.join(home_dir+'Mask_RCNN-master/'),
               os.path.join(home_dir+'cocoapi/PythonAPI/'),
               os.path.join(home_dir+'cocoapi/PythonAPI/pycocotools/'),
               os.path.join(home_dir+'Mask_RCNN-master/samples/driving'))
for p in module_path:
  if p not in sys.path:
    sys.path.append(p)

os.chdir(os.path.join(home_dir, 'Mask_RCNN-master/samples/driving'))

# Root directory of the project
ROOT_DIR = os.path.join(home_dir, 'Mask_RCNN-master/samples/driving') #os.getcwd()
if ROOT_DIR.endswith("samples/driving"):
    # Go up two levels to the repo root
    ROOT_DIR = os.path.dirname(os.path.dirname(ROOT_DIR))

# Import Mask RCNN
sys.path.append(ROOT_DIR)
import utils
import visualize
from visualize import display_images
import model as modellib
from model import log

import wad_tools

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to WAD trained weights
WAD_WEIGHTS_PATH = os.path.join(ROOT_DIR, "samples/driving", "mask_rcnn_wad_0069.h5")  # TODO: update this path
COCO_WEIGHTS_PATH = None
print('ROOT_DIR',ROOT_DIR)


# ## Configurations
config = wad_tools.WADConfig()
WAD_DIR = os.path.join(home_dir, "data/wad")

# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024

config = InferenceConfig()
config.display()


# Device to load the neural network on.
# Useful if you're training a model on the same
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"


# ## Load Test Dataset

# Load validation dataset
labels_file = "/home/paperspace/data/wad/wad_labels.csv"
dataset = wad_tools.WADDataset(labels_file, config.DATASET_DICT, ignore_cats=config.IGNORE_CATS_LIST)
dataset.load_wad(WAD_DIR, "test")

# Must call before using the dataset
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))


# ## Load Model
# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)

# Set path to WAD model weights file
weights_path = WAD_WEIGHTS_PATH

# Or, load the last model you trained
#weights_path = model.find_last()[1]

# Load weights
print("Loading weights ", weights_path)
if weights_path == COCO_WEIGHTS_PATH:
    model.load_weights(weights_path, by_name=True, exclude=[
                "mrcnn_class_logits", "mrcnn_bbox_fc",
                "mrcnn_bbox", "mrcnn_mask"])
else:
    model.load_weights(weights_path, by_name=True)

# ## Get all Test Predictions
preds_test=np.array([dict() for _ in dataset.image_ids])

for i, image_id in tqdm(enumerate(dataset.image_ids), total=len(dataset.image_ids), unit="images"):
  image = dataset.load_image(image_id)
  info = dataset.image_info[image_id]
  #print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
  #                                     dataset.image_reference(image_id)))

  # Run object detection
  results = model.detect([image], verbose=0)
  preds_test[i]['id']=info["id"]
  preds_test[i]['image']=image
  mask = (np.sum(results[0]["masks"], -1, keepdims=True) >= 1)
  preds_test[i]['pred']=np.squeeze(mask)
  #print(preds_test[i]['pred'].shape)

np.save('preds_test', preds_test)


# # Encode and submit our results
#

from skimage.morphology import label
def rle_encoding(x):
    """ Run-length encoding from
    https://www.kaggle.com/kmader/opencv-hog-submission/code based on
    https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
    Modified by Konstantin, https://www.kaggle.com/lopuhin
    """
    assert x.dtype == np.bool
    dots = np.where(x.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.append([b, 0])
        run_lengths[-1][1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)


# Let's iterate over the test IDs and generate run-length encodings
# for each seperate mask identified by skimage ...

new_test_ids = []
rles = []
for n, id_ in enumerate(dataset.image_ids):
    print("shape:", n, preds_test[n]["pred"].shape, preds_test[n]["image"].shape)
    #imshow((X_test[n]*tstd+tmean)*255)
    #plt.show()
    #imshow(preds_test_upsampled[n])
    #plt.show()
    rle = list(prob_to_rles(preds_test[n]["pred"]))
    rles.extend(rle)
    new_test_ids.extend([preds_test[n]["id"]] * len(rle))


# ... and then finally create our submission!

print(rles[1])

from google.colab import files
import pandas as pd
# Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))

#save as a csv file
sub.to_csv(os.path.join(ROOT_DIR, 'samples/wad', 'wad_mask_rcnn_test.csv'), index=False)
