
sherif - v4 2023-05-08 1:12pm
==============================

This dataset was exported via roboflow.com on October 29, 2025 at 3:36 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 890 images.
Phone-seatbelt are annotated in YOLO v5 PyTorch format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 1 versions of each source image:

The following transformations were applied to the bounding boxes of each image:
* Randomly crop between 28 and 75 percent of the bounding box


