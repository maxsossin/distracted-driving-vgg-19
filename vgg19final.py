import os
from shutil import copy
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from math import ceil
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import roc_curve, auc
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.backend import sigmoid
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
from IPython.display import display
from PIL import Image

target_size = 224, 224
batch_size = 2
class_labels_encoded = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
class_labels = ['safe_driving', 'texting_right', 'talking_on_phone_right', 'texting_left', 'talking_on_phone_left',
                'operating_radio', 'drinking', 'reaching_behind', 'doing_hair_makeup', 'talking_to_passanger']
num_classes = len(class_labels)
#Create the swish custom activation function
def swish(x):
    return (x * sigmoid(x))
get_custom_objects().update({'swish': Activation(swish)})
#Creates top fully connected layers for VGG-19
def setup_fc_layers(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(4096, activation='swish'))
    model.add(Dense(4096, activation='swish'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))
    return model
NUM_CLASSES = 10
data_path = './imgs/train/'
#Rescale image to account for RGB
datagen = ImageDataGenerator(rescale=1. / 255)
#Split the data from kaggle into test train and validate sets
for i in range(NUM_CLASSES):
    curr_dir_path = data_path + 'c' + str(i) + '/'
    xtrain = labels = os.listdir(curr_dir_path)
    print(curr_dir_path)
    print(xtrain)
    x, x_test, y, y_test = train_test_split(xtrain, labels, test_size=0.2, train_size=0.8)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.25, train_size=0.75)
    os.makedirs('train/' + 'c' + str(i) + '/', exist_ok=True)
    os.makedirs('test/' + 'c' + str(i) + '/', exist_ok=True)
    os.makedirs('validation/' + 'c' + str(i) + '/', exist_ok=True)
    for x in x_train:
        if (not os.path.exists('train/' + 'c' + str(i) + '/' + x)):
            copy(data_path + 'c' + str(i) + '/' + x, 'train/' + 'c' + str(i) + '/' + x)

    for x in x_test:
        if (not os.path.exists('test/' + 'c' + str(i) + '/' + x)):
            copy(data_path + 'c' + str(i) + '/' + x, 'test/' + 'c' + str(i) + '/' + x)

    for x in x_val:
        if (not os.path.exists('validation/' + 'c' + str(i) + '/' + x)):
            copy(data_path + 'c' + str(i) + '/' + x, 'validation/' + 'c' + str(i) + '/' + x)
#Create the VGG19 model with imagenet weights
model = VGG19(include_top=False, weights='imagenet')
#Get the unlabeled train, validate and test data for feature extraction
train_generator = datagen.flow_from_directory(
    './train/',
    target_size=target_size,
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)
val_generator = datagen.flow_from_directory(
    './validation/',
    target_size=target_size,
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)
test_generator = datagen.flow_from_directory(
    './test/',
    target_size=target_size,
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)
# #Extract features and save them externally

train_features = model.predict_generator(train_generator, steps=ceil(len(train_generator.filenames) / batch_size), verbose=1)
np.save('saved/train_features.npy', train_features)
val_features = model.predict_generator(val_generator, steps=ceil(len(val_generator.filenames) / batch_size), verbose=1)
np.save('saved/val_features.npy', val_features)
test_features = model.predict_generator(test_generator, steps=ceil(len(test_generator.filenames) / batch_size), verbose=1)
np.save('saved/test_features.npy', test_features)
epochs = 50
#Generate labelled data for training
datagen = ImageDataGenerator(rescale=1. / 225)
train_generator = datagen.flow_from_directory(
    './train/',
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)
val_generator = datagen.flow_from_directory(
    './validation/',
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)
#Load deep features and do onehot encoding
train_data = np.load('saved/train_features.npy')
train_labels_onehot = to_categorical(train_generator.classes, num_classes=num_classes)
val_data = np.load('saved/val_features.npy')
val_labels_onehot = to_categorical(val_generator.classes, num_classes=num_classes)
#Compile top fully connected layers
model = setup_fc_layers(train_data.shape[1:])
model.compile(optimizer="adadelta", loss="categorical_crossentropy", metrics=["accuracy"])
#Save best weights externally
checkpoint_callback = ModelCheckpoint(
    "saved/best_weights.h5",
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max')
#Stop model early if accuracy doesn't change after 3 epochs
early_stop_callback = EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    mode='max')
callbacks_list = [checkpoint_callback, early_stop_callback]
#Train the top model layers on the features extracted previously
history = model.fit(
    train_data,
    train_labels_onehot,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(val_data, val_labels_onehot),
    callbacks=callbacks_list)
#Generate labelled testdata
datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_generator = datagen.flow_from_directory(
    './test/',
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)
#Load features extracted previously and get onehot labels
test_data = np.load('saved/test_features.npy')
test_labels_onehot = to_categorical(test_generator.classes, num_classes=num_classes)  # class number in onehot
#Load wieghts saved during training and compile model
model = setup_fc_layers(test_data.shape[1:])
model.load_weights("saved/best_weights.h5")
model.compile(optimizer="adadelta", loss="categorical_crossentropy", metrics=["accuracy"])
#Get results of the model on test data
predicted = np.argmax(model.predict(test_data), axis=-1)
loss, acc = model.evaluate(test_data, test_labels_onehot, batch_size=batch_size, verbose=1)
print("loss: ", loss)
print("accuracy: {:8f}%".format(acc * 100))

#Create the confusion matrix and plot it
def display_c_matrix(y_true, y_pred):
    c_matrix = confusion_matrix(y_true, y_pred)
    title = 'Confusion matrix'
    c_matrix = c_matrix.astype('float') / c_matrix.sum(axis=1)[:, np.newaxis]
    plt.figure()
    plt.imshow(c_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_labels_encoded))
    plt.xticks(tick_marks, class_labels_encoded, rotation=0)
    plt.yticks(tick_marks, class_labels_encoded)
    for i, j in itertools.product(range(c_matrix.shape[0]), range(c_matrix.shape[1])):
        plt.text(j, i, format(c_matrix[i, j], '.2f'),
                 horizontalalignment='center',
                 color='white' if c_matrix[i, j] > (c_matrix.max() / 2) else 'black')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
#Create the roc curve and plot it, based on turtorial found here https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
def display_roc_curve(y_true, y_pred):
    # calculate roc and auc
    fp_rate = dict()
    tp_rate = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fp_rate[i], tp_rate[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fp_rate[i], tp_rate[i])
    #plot the roc curve for the model
    cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, num_classes))
    for i, color in zip(range(num_classes), colors):
        plt.plot(fp_rate[i], tp_rate[i], lw=2, c=color,
                 label='c{0} (auc = {1:0.2f})'.format(i, roc_auc[i]))
    # plot the roc curve for random guesses
    plt.plot([0, 1], [0, 1], 'k--', color='salmon', lw=2, label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid()
    plt.show()

display_c_matrix(test_generator.classes, predicted)
#Convert predicted results to onehot 
predicted_onehot = to_categorical(predicted, num_classes=num_classes)
display_roc_curve(test_labels_onehot, predicted_onehot)