# **Behavioral Cloning**

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model.png "Model Visualization"
[image2]: ./examples/data1.png "data1"
[image3]: ./examples/flipped.png "Flipped Image"
[image4]: ./examples/nvidia.png "CNN"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on the NVIDIA model architecture discussed in " End to end learning for self driving cars". It consists of a convolutional neural network with 3 5x5 convolutional layer with stride 2x2, 2 non-strided 3x3 convolutional layer and 3 fully connected layers. The convolutional layers have RELU activations to introduce non-linearity. The fully connected layers use ELU activations.

The data is converted to YUV color space and cropped to a size of 200x66. It is normalized in the model using a Keras lambda layer.

![alt text][image4]

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I used the NVIDIA architecture described in the publication "End to end learning for self driving cars". The authors describe that the the primary motivation for their work was to avoid the need to recognize specific human-designated features, such as lane markings, guard rails, or other cars,
and to avoid having to create a collection of “if, then, else” rules, based on observation of these features.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model by adding dropout layers to the first two connected layers. This reduced the overfitting and the validation loss reduced for 2 more epochs.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, like during steep turn before the bridge and also veered right after bridge crossing. To improve the driving behavior in these cases, I collected some more training data by driving in the center near these regions.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture is the NVIDIA architecture with dropout layers added for the fully connected layers.

Here is a visualization of the architecture.

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

I started off with the udacity driving data which included center driving with about 20K images from the 3 cameras. Here is an example of the images from the 3 cameras.

![alt text][image2]

I then recorded some additional center driving around steep curve before the bridge and after the bridge. The open space after bridge confused the car into thinking it was a lane. This additional data helped to correct the driving in that area.

To augment the data sat, I also flipped images and angles thinking that this would help the network generalize. For example, here is an image that has then been flipped:

![alt text][image3]

After the collection process, I had 26.5K data points. I then preprocessed this data by
* Convert image to YUV colorspace.
* Crop the top 60 rows and bottom 25 rows.
* Resize image from 320 x 75 to 200 x 66

I finally randomly shuffled the data set and put 20% of the data into a validation set.

The batch generator also
* randomly selected the image from either the left, center or right cameras. I have 50% chance to the center camera and 25% to the other two.
* randomly flipped the image.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by validation loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.
