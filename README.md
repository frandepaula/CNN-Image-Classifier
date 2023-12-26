# 🐶 Dog Breed Image Classification 

Developed as the final submission for a Machine Learning class, this project focuses on classifying dog breeds through sophisticated computer vision methods. The primary focus is on comparing two advanced convolutional neural network (CNN) architectures: Xception and Inception v3. 

## CNN architectures:

* Xception: deep learning architecture known for its extreme inception-style design. It employs depthwise separable convolutions, breaking down the standard convolutional layer into separate spatial and depthwise convolutions. This enhances efficiency and reduces the number of parameters, making it well-suited for image classification tasks.

* Inception v3: developed by Google, is part of the Inception family of neural network architectures. It introduces the concept of inception modules, which use multiple filter sizes simultaneously to capture features at different scales. This enables the network to learn hierarchical representations and improves its ability to recognize complex patterns in images.


## Dataset

The Stanford Dogs dataset contains images of 120 dog breeds from around the world. This dataset was constructed using images and annotations from ImageNet for the high-resolution image categorization task. This dataset includes:

* Number of categories: 120
* Number of images: 20,580
* Annotations: Class labels, Bounding boxes

#### Source: http://vision.stanford.edu/aditya86/ImageNetDogs/

## Packages

The scripts are written in Python and can be executed from the command line. Running the tool requires Python 3 support. Additionally, the following libraries must be installed locally:

* Keras
* Split_folders
* ImageDataGenerator from keras.preprocessing.image
* Sequential from keras.models
* Dense, Dropout, Flatten from keras.layers
* Conv2D, MaxPooling2D from keras.layers
* models from keras
* layers from keras
* vgg16 from keras.applications.vgg16
* optimizers from keras
* plot_model from keras.utils
* load_model from keras.models
* xception from keras.applications.xception
* to_categorical from keras.utils.np_utils
* confusion_matrix from sklearn.metrics
* load_model from keras.models
* matplotlib.pyplot
* itertools
* sklearn.metrics
* numpy
* pandas
