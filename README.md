# WiDS Datathon 2019
Predict Oil Palm Plantation from Satellite Images

Image classification problem: train a model that takes as input a satellite image and outputs a prediction of how likely it is that the image contains an oil palm plantation. The model is used to make prediction on the unlabeled images in test set. This is a [WiDS Datathon 2019](https://www.kaggle.com/c/widsdatathon2019) in Kaggle. 

The training set images are stored in folder `train_images` and the test images are stored in folder `leaderboard_test_data`, and `leaderboard_holdout_data`. The label is in the file `traininglabels.csv` with label 1 means the image has oil palm.
 
* Libraries: fastai, pytorch libraries, opencv
* Hardwares: Nvidia GTX1060 3 GB
* Split image into train, validation and test sets
* Workwith Imbalance class: only 6% of the images belong to a second class (with oilpalm). Solve by create an augmented images from the training set. Details in this [notebook](https://github.com/worasom/WiDS_Datathon_2019/blob/master/generate_augmented_images.ipynb)
*	Transfer learning using pretrained models  
* Explore different CNN architectures: resnet34, resnext201, dn201 in this [notebook](https://github.com/worasom/WiDS_Datathon_2019/blob/master/oil_palm_images-arch_survey.ipynb)
* Submit different prediction to kaggle to verify correlation between the hold out set and the internal test/validation sets [notebook](https://github.com/worasom/WiDS_Datathon_2019/blob/master/oil_palm_images-arch_survey.ipynb).
* Pretrained neural network using previous Kaggle competition in this[notebook](https://github.com/worasom/WiDS_Datathon_2019/blob/master/pretrain_planet.ipynb) and use the trained weight for this problem in this [notebook](https://github.com/worasom/WiDS_Datathon_2019/blob/master/oil_palm_images-trans.ipynb).
*	Achieve 95% accuracy for the Kaggle hold out dataset (151 on the leaderboard) 

I learn a lot from working on this project. First, creating augmented images is the key to work with imbalance dataset. Second, it looks like deep network perform better on this dataset, but once picking a deep NN, it does not matter if it's resnext101 or restnet152 or dn201. Third, I learned how to transfer learning weights from other images to this set. Although the accuracy was not better, but it was a very good experience. 