# WiDS Datathon 2019
Predict Oil Palm Plantation from Satellite Images

Image classification problem: train a model that takes as input a satellite image and outputs a prediction of how likely it is that the image contains an oil palm plantation. The model is used to make prediction on the unlabeled images in test set. This is a [WiDS Datathon 2019](https://www.kaggle.com/c/widsdatathon2019) in Kaggle. 

The training set images are stored in folder `train_images` and the test images are stored in folder `leaderboard_test_data`, and `leaderboard_holdout_data`. The label is in the file `traininglabels.csv` with label 1 means the image has oil palm.
 
*	libraries fastai, pytorch libraries, opencv
* split image into train, validation and test sets
* Imbalance class: only 6% of the images belong to a second class (with oilpalm). Solve by create an augmented images from the training set
*	Transfer learning using pretrained models  
* Find different CNN model architectures: resnet34, resnext201, dn201
* Submit different prediction to kaggle to verify correlation between the hold out set and the internal test/validation sets.
*	Achieve 95% accuracy (about 151 on the leaderboard) 
