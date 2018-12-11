# Photo Uncrop

This repository contains code and usage instructions for our COMS 4731 project.

## Description

## Dependencies
We developed the code using Tensorflow for python 2 in Anaconda. Also the following packages need to be installed:

* OpenCV 
* Pandas
* Numpy
* Ipdb
* Scikit-Image

For both training and testing, we used a Google cloud instance with an NVIDIA Tesla P100 GPU.

## Dataset
We used the following 10 classes of outdoor scenes from the [Places dataset](http://places2.csail.mit.edu/index.html): Butte, corn-field, desert, desert-road, farm, field-road, hay-field, mountain-path, pasture, sky. 
</br>
To prepare the dataset, download the train and validation set under "Small images (256 * 256)" from http://places2.csail.mit.edu/download.html. Extract into folders and remove all classes other than the 10 required ones. We use the train set for training and the validation set for testing.

## Usage instructions
All code is housed in the `src/` folder.
### Training
To train the model with mask reconstruction loss:
```
python train_mask.py
```

To train the model with unmask reconstruction loss:
```
python train_unmask.py
```
### Testing
To generate test results using a trained model


### Post-processing

### Evaluation

## Pre-trained model

## References


