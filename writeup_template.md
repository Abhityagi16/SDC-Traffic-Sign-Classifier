# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/predictions.png "Predictions on New Images"
[image4]: ./test_images/0rigtoffway_atnextintersection_32x32x3.jpg "Traffic Sign 1"
[image5]: ./test_images/1priority_road_32x32x3.jpg "Traffic Sign 2"
[image6]: ./test_images/2_yield.jpg "Traffic Sign 3"
[image7]: ./test_images/3_noentry_32x32x3.jpg "Traffic Sign 4"
[image8]: ./test_images/4wildanimalscrossing_32x32x3.jpg "Traffic Sign 5"
[image9]: ./test_images/5turn_left_ahead.jpg "Traffic Sign 6"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how many samples are contained in the training set per label.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because identifying objects in image doesn't get affected by color and reducing an image to grayscale significantly reduce training time.

As second and last step, I normalized the image data 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The model architecture is based on the LeNet model architecture. I added dropout layers before each fully connected layer in order to prevent overfitting. My final model consisted of the following layers:


| Layer                  |     Description                                |
|------------------------|------------------------------------------------|
| Input                  | 32x32x1 gray scale image                       |
| Convolution 5x5        | 1x1 stride, valid padding, outputs 28x28x6     |
| RELU                   |                                                |
| Max pooling            | 2x2 stride,  outputs 14x14x6                   |
| Convolution 5x5        | 1x1 stride, valid padding, outputs 10x10x16    |
| RELU                   |                                                |
| Max pooling            | 2x2 stride,  outputs 5x5x16                    |
| Flatten                | outputs 400                                    |
| **Dropout**            |                                                |
| Fully connected        | outputs 120                                    |
| RELU                   |                                                |
| **Dropout**            |                                                |
| Fully connected        | outputs 84                                     |
| RELU                   |                                                |
| **Dropout**            |                                                |
| Fully connected        | outputs 43                                     |
| Softmax                |                                                |



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer and the following hyperparameters:
* batch size: 128
* number of epochs: 150
* learning rate: 0.0006
* Variables were initialized using the truncated normal distribution with mu = 0.0 and sigma = 0.1
* keep probalbility of the dropout layer: 0.5


My final model results were:
* training set accuracy of 99.5%
* validation set accuracy of 96.6%
* test set accuracy of 94.7%


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I used an iterative approach for the optimization of validation accuracy:

1. As an initial model architecture the original LeNet model from the course was chosen. In order to tailor the architecture for the traffic sign classifier usecase I adapted the input so that it accepts the colow images from the training set with shape (32,32,3) and I modified the number of outputs so that it fits to the 43 unique labels in the training set. The training accuracy was **83.5%** and my test traffic sign "pedestrians" was not correctly classified. 
  (used hyper parameters: EPOCHS=10, BATCH_SIZE=128, learning_rate = 0,001, mu = 0, sigma = 0.1) 

1. After adding the grayscaling preprocessing the validation accuracy increased to **91%** 
   (hyperparameter unmodified)

1. The additional normalization of the training and validation data resulted in a minor increase of validation accuracy: **91.8%** (hyperparameter unmodified)

1. reduced learning rate and increased number of epochs. validation accuracy = **94%** 
   (EPOCHS = 30, BATCH_SIZE = 128, rate = 0,0007, mu = 0, sigma = 0.1)

1. overfitting. added dropout layer after relu of final fully connected layer: validation accuracy = **94,7%** 
   (EPOCHS = 30, BATCH_SIZE = 128, rate = 0,0007, mu = 0, sigma = 0.1)

1. still overfitting. added dropout after relu of first fully connected layer. Overfitting reduced but still not good

1. added dropout before validation accuracy = 0.953 validation accuracy = **95,3%** 
   (EPOCHS = 50, BATCH_SIZE = 128, rate = 0,0007, mu = 0, sigma = 0.1)

1. further reduction of learning rate and increase of epochs. validation accuracy = **97,5%** 
   (EPOCHS = 150, BATCH_SIZE = 128, rate = 0,0006, mu = 0, sigma = 0.1)

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]

The "right-of-way at the next intersection" might be difficult to classify because the triangular shape is similiar to several other signs in the training set (e.g. "Child crossing" or "Slippery Road"). 
Additionally, the "Stop" sign might be confused with the "No entry" sign because both signs have a round shape and a big red area.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.7%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The detailed probabilities and examples of the top five softmax predictions are given in the this image.
![alt text][image2]
