# Car Color Spash Example

In this example we experiment with using Mask-RCNN to detect cars and other objects from the CVPR 1018 WAD Kaggle dataset (https://www.kaggle.com/c/cvpr-2018-autonomous-driving). This is currently under construction. The example is based on Waleedka's Baloons example.

Below are two examples of cars and trucks detected in teh image
![Car Detection](https://github.com/ReemHal/Mask_RCNN_Nucli/blob/master/samples/driving/figures/car_detection_2.png)
![Car Segmentation](https://github.com/ReemHal/Mask_RCNN_Nucli/blob/master/samples/driving/figures/car_detection.png)

## Installation
[You can find the Mask-RCNN code and baloon example here: (https://github.com/matterport/Mask_RCNN/releases)]
1. Download `mask_rcnn_balloon.h5`. Save it in the root directory of the repo (the `mask_rcnn` directory).
2. Download Kaggle's CVPR 1018 WAD dataset (https://www.kaggle.com/c/cvpr-2018-autonomous-driving). Expand it such that it's in the path `mask_rcnn/datasets/driving/`.

## Apply color splash using the provided weights
Apply splash effect on an image:

```bash
python3 driving.py splash --weights=/path/to/mask_rcnn/mask_rcnn_balloon.h5 --image=<file name or URL>
```

## Run Jupyter notebooks
Open the `inspect_driving_data.ipynb` or `inspect_driving_model.ipynb` Jupter notebooks. You can use these notebooks to explore the dataset and run through the detection pipelie step by step.

## Train the Balloon model

Train a new model starting from pre-trained COCO weights
```
python3 driving.py train --dataset=/path/to/driving/dataset --weights=coco
```

Resume training a model that you had trained earlier
```
python3 driving.py train --dataset=/path/to/balloon/dataset --weights=last
```

Train a new model starting from ImageNet weights
```
python3 driving.py train --dataset=/path/to/balloon/dataset --weights=imagenet
```

