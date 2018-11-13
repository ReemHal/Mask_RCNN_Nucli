# Detect Nucli Example

This is an example that uses Mask-RCNN to identify cell nucli from Kaggle's Data Science Bowl 2018 (https://www.kaggle.com/c/data-science-bowl-2018). The example follows the Balloons example by Waleedka (https://github.com/waleedka) but on images only. Below are examples of the nuclei highlighted with a color splash and identified with a bounding box.

![Nuclei shown with color splash](https://github.com/ReemHal/Mask_RCNN_Private/blob/master/samples/nucli/figures/color_splash_nucli.png)

![Nuclei Bounding Box](https://github.com/ReemHal/Mask_RCNN_Private/blob/master/samples/nucli/figures/Bounding_Box_nucli.png)

Update: Please note this is not the same as the recently uploaded Nucleus example by Waleedka.

## Installation

1. Download `mask_rcnn_balloon.h5` ((https://github.com/matterport/Mask_RCNN/releases). Save it in the root directory of the repo (the `mask_rcnn` directory).
2. Download Kaggle's nucli dataset from https://www.kaggle.com/c/data-science-bowl-2018/data. Expand it such that it's in the path `mask_rcnn/datasets/nucli/`.

## Apply color splash using the provided weights

Apply splash effect on an image:

```bash
python3 nucli.py splash --weights=/path/to/mask_rcnn/mask_rcnn_balloon.h5 --image=<file name or URL>
```

## Run Jupyter notebooks
Open the `inspect_nucli_data.ipynb` or `inspect_nucli_model.ipynb` Jupter notebooks. You can use these notebooks to explore the dataset and run through the detection pipelie step by step.

## Train the Balloon model

Train a new model starting from pre-trained COCO weights
```
python3 nucli.py train --dataset=/path/to/nucli/dataset --weights=coco
```

Resume training a model that you had trained earlier
```
python3 nucli.py train --dataset=/path/to/nucli/dataset --weights=last
```

Train a new model starting from ImageNet weights
```
python3 nucli.py train --dataset=/path/to/nucli/dataset --weights=imagenet
```
