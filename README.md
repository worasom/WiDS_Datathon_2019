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
*	Achieve 99.4% accuracy for the Kaggle hold out dataset (113 on the leaderboard) 

I learn a lot from working on this project. First, creating augmented images is the key to work with imbalance dataset. Second, it looks like deep network perform better on this dataset. Third, I learned how to transfer learning weights from other images to this set. Although the accuracy was not better, but it was a very good experience. 

# Exploratory Data Analysis

Examples of images. 

![](https://github.com/worasom/WiDS_Datathon_2019/blob/master/figgit/fig1.png)

Image size are 256 x 256 pixels.  


In summary, we have 15244 training images, and will allocate 1524(10%) as a test and as 2744 validation(20% of the remaining), leaving 10976 for the training set. There are two hold out sets: leader_board_test folder has 4356 files and learder_board_holdout_data has 2178 files. The internal test images are copied to folder test, and the train image list is in the 'train_val.csv'.

About 6% of the images have oil palm. The training set should have about 658 images with oil palm. The images are 256 x 256 pixels and have 3 channels. Baseline: if a model predicts all images to have no oil palm will get 94% accuracy and 2 log_loss. Public leader board is 99.957%.

Leaderboard test images are in the same folder as the internal test images.


# Imbalance Training Data 


This dataset has unbalance dataset, only 6% of images have oil palm. To solve this, I will add augmented images this class in the training folder.

1. Define functions to generate augmented images:

- rotate_cv return a rotated image. I use this to create 7 more images each with multiple of 45 deg rotation.
- stretch_cv return 3 images with a 1.01 horizontal or vertical zoom or both.
- I will get (7+3) x 672 = 6720 images
- Randomly select half of these images to put into either lighting or blur. This procedure should give me
- At the end, I will have ((7+3) x 672 + 3360 about 10080 images or more (if perform randomness two times will generate more images)
- Randomly shuffle the images and include 8000 images into a training label csv file called 'train_val_augment.csv' with these filenames and label 1. Note that the images are in a separate folder)

This is done in [notebook](https://github.com/worasom/WiDS_Datathon_2019/blob/master/generate_augmented_images.ipynb). Here, I am going to show example of the augmented images.

Examples of images. 

![](https://github.com/worasom/WiDS_Datathon_2019/blob/master/figgit/fig2.png)


# Try Different Architectures and Pick one

I need to pick the architecture. We have the following pretained models: resnet18, resnet34, resnet50, resnet101, resnet152, vgg16, vgg19, resnext50, resnext101, resnext101_64, wrn, inceptionresnet_2, inception_4, dn121, dn161, dn169, dn201.

Use small learning rate and leave this turn over night. 

Furthermore, I need to verify if the internal test set is similar to the kaggle test set. If they are similar the accuracy scores for the internal test set should be the same. This is done by submitting results from 3 - 5 models and plot the internal accuracy vs submitted accuracy. 

The result is shown here.

![](https://github.com/worasom/WiDS_Datathon_2019/blob/master/figgit/fig3.png)

Higher validation score correlated with higher Kaggle score. This means the validation set is pretty similar to the hold out set.

![](https://github.com/worasom/WiDS_Datathon_2019/blob/master/figgit/fig4.png)

Test set is not that similar to the hold out. 
![](https://github.com/worasom/WiDS_Datathon_2019/blob/master/figgit/fig5.png)

Now I can rank the model by the validation score, pick the best model and optimized them. 


# Summary and Future Work

In summary, there seems to be a good correlation between the validation set and the hold out set. Therefore, I can use the validation accuracy for choosing a model. The best architecture is resnext101, which gives accuracy of 99.38% on Kaggle hold out set (113 position on the leaderboard). The other two models, dn201, and resnet152 are also good.

Using a train the weights from another set of satellite images. Use the pretrained weight on this dataset, does not give better accuracy. Therefore, to improve the accuracy, I think the way to go is to generate more augmented images for both classes.

