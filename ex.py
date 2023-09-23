"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
#import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from utils import split_train_dev_test, predict_and_eval, train_model, tune_hparams
from itertools import product
from skimage.transform import resize

gamma = [0.001, 0.01, 0.1, 1, 10, 100]
C_range = [0.1, 1, 2, 5, 10]
test_range = [0.1, 0.2, 0.3]
dev_range = [0.1, 0.2, 0.3]

###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.

digits = datasets.load_digits()

# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, label in zip(axes, digits.images, digits.target):
#     ax.set_axis_off()
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
#     ax.set_title("Training: %i" % label)

###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.

# flatten the images
n_samples = len(digits.images)
r_digits = resize(digits.images, (digits.images.shape[0] // 4, digits.images.shape[1] // 4),
                       anti_aliasing=True)

data = r_digits.images.reshape((n_samples, -1))

height, width, _ = r_digits.images.shape

print("height={} width={}".format(height, width))

#hyper parameter tuning
#h_parameters = dict(product(gamma, C_range,repeat=1))
h_parameters=list(product(gamma, C_range))

dataset_combination = list(product(test_range, dev_range))

for dataset in dataset_combination:
    t_size = dataset[0]
    d_size = dataset[1]
    train_size = 1 - t_size - d_size   

    # Split data into 50% train and 30% test and 20% Validate subsets
    X_train, X_dev, X_test, y_train, y_dev, y_test  = split_train_dev_test(
        data, digits.target, test_size=t_size, dev_size=d_size
    )

    optimal_gamma, optimal_C, optimal_model, optimal_accuracy = tune_hparams(X_train,y_train,X_dev,y_dev,h_parameters)

    print("optimal Gamma value={} and optimal C value={}".format(optimal_gamma,optimal_C))

    #Either we store model in loop or we train it again based on size of model
    #clf = train_model(X_train, y_train, {'gamma':optimal_gamma, 'C':optimal_C})

    #Train accuracy
    train_accuracy = predict_and_eval(optimal_model, X_train, y_train)

    # Predict the value of the digit on the test subset
    test_accuracy = predict_and_eval(optimal_model, X_test, y_test)

    train_sample = len(X_train)
    dev_sample = len(X_dev)
    test_sample = len(X_test)

    print("Training sample={} dev Samples={} test Samples={}".format(train_sample, dev_sample, test_sample))

    print("images 4*4, train_size={} train_accuracy={}, dev_size={} dev_accuracy={}, test_size={} test_accuracy={}".format(train_size,train_accuracy,d_size,optimal_accuracy,t_size,test_accuracy))

###############################################################################
# Below we visualize the first 4 test samples and show their predicted
# digit value in the title.

# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, prediction in zip(axes, X_test, predicted):
#     ax.set_axis_off()
#     image = image.reshape(8, 8)
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
#     ax.set_title(f"Prediction: {prediction}")

###############################################################################
