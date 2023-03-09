# Image-segmentation-API

## Overview
In this project I trained an image segmentation model with tensor flow to detect features from a picture for automated vehicle. The model was then deployed on Heroku using Flask API. The pipeline was designed with Azure ML.

## Data
The famous dataset [Cityscape](https://www.cityscapes-dataset.com/dataset-overview/) was used for this project.

## Design and training
Two designs were tested : U-net developped in 2015 for biomedical images ([article](https://heartbeat.comet.ml/deep-learning-for-image-segmentation-u-net-architecture-ff17f6e4c1cf)) and Deeplab from Google ([paperswithcode](https://paperswithcode.com/method/deeplab#:~:text=DeepLab%20is%20a%20semantic%20segmentation,we%20obtain%20the%20final%20predictions.)).

## Data Augmentation
I used the Augmentor package for for Data Augmentation. The following features were applied with different magnitudes:
* Roatation
* Random zoom
* Perspective skewing left, right, top, bottom and corner. (transform the image so that it appears that you are looking at it from a different angle)
* Elastic distortion (random distortions while maintaining the image's aspect ratio)  
[Augmentator](https://augmentor.readthedocs.io/en/stable/)

## Hyper parameters 
Baseline was established using :
optimizer Adam
learning rate = 0.0001
Batch size = 32

I studied the impact of data augmentation, different batch size and learning rate. Due to limited computation capacity the test were limited to only a few values.

I used TensorBoard to monitor training.  
[TensorBoard](https://www.tensorflow.org/tensorboard?hl=fr)

## Deployment
Using [Flask](https://flask.palletsprojects.com/en/2.2.x/) to build the API and [Heroku](https://www.heroku.com) (when it was still free) for deployment.
