# FashionMNIST

##Overview
This is a dataset from Kaggle that contains grayscale images of various clothes items (boots, sandals, shirts, etc) of 10 different classes. I built a classification model in Tensorflow.

##Architecture
An image is taken and reshaped into a 1D vector and normalized. I started with a simple model with 2 hidden layers, and experimented with number and size of hidden layers, gradually increasing the model size to get me to the desired accuracy. 

##Data cleaning
Not a problem here; this data set is well known for being clean and useful for trying things out.

##Training
I experimented with various model sizes and various training length; small model obviously maxed out and on a long training started to overfit.

##Results
I ended up building a model with 4 hidden layers to get me ~89 percent prediction on the testing set. There is still reserve to bump up predictive performance a few more percentage points by making the model bigger, given that the training set performance is 92.5 percent and can be made better and the data set was clearly responsive to a more sophisticated model.