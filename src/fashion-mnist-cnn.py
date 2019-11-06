import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import keras
from keras.models import Sequential, InputLayer
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.datasets import fashion_mnist
from keras import backend as K
from keras.layers import Activation
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split  
from keras.models import load_model
import argparse

def load_data(path):
    df_train = pd.read_csv(path + 'fashion-mnist_train.csv')
    df_test = pd.read_csv(path + 'fashion-mnist_test.csv')

    df_features = df_train.iloc[:, 1:785]
    df_label = df_train.iloc[:, 0]

    x_test = df_test.iloc[:, 1:785]
    y_test = df_test.iloc[:, 0]

    return df_features, df_label, x_test, y_test

def preprocess_data(x_train, y_train, x_test, y_test, num_classes, input_shape):

    img_rows, img_cols = input_shape

    x_train = np.array(x_train)
    x_test = np.array(x_test)

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows* img_cols)
        input_shape = (img_rows* img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows* img_cols)


    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, 
                                                test_size = 0.2,
                                                random_state = 1212)
    
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)

    return x_train, y_train, x_val, y_val, x_test, y_test


if __name__ == '__main__':
    # Call argument for command
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default='/home/hung/study/CV_APPLICATIONS/Fashion_MNIST/fashionmnist/', type=str)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    args = parser.parse_args()


    # Load fasion-mnist data
    x_train, y_train, x_test, y_test = load_data(args.path)
    x_train, y_train, x_val, y_val, x_test, y_test = preprocess_data(x_train, y_train, x_test, y_test, 
                                                                     num_classes=10, input_shape=(28,28))
    print("X_train's shape: " + str(x_train.shape))
    print("X_val's shape: " + str(x_val.shape))


    # Create model
    model = Sequential()

    # input: 28x28 images with 1 channels -> (28, 28, 1) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(InputLayer(input_shape = (28, 28, 1)))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))

    model.add(BatchNormalization())
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()
    

    # Train the model
    model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs)

    # Evaluate model on training set
    score_train = model.evaluate(x_train, y_train, batch_size=64)
    print('train_loss: %.5f - train_acc: %.2f%%' % (score_train[0], score_train[1]*100))

    # Evaluate model on valid set
    score_val = model.evaluate(x_val, y_val, batch_size=64)
    print('val_loss: %.5f - val_acc: %.2f%%' % (score_val[0], score_val[1]*100))

    # Evaluate model on test set
    score_test = model.evaluate(x_test, y_test, batch_size=64)
    print('test_loss: %.5f - test_acc: %.2f%%' % (score_test[0], score_test[1]*100))

    # Predict for test dataset
    y_pred = model.predict(x_test)

    # Convert test to labels 
    y_hat = np.argmax(y_pred, axis=1)
    labels_text = ['T-shirt/top',
                    'Trouser/pants',
                    'Pullover shirt',
                    'Dress',
                    'Coat',
                    'Sandal',
                    'Shirt',
                    'Sneaker',
                    'Bag',
                    'Ankle boot']

    # Show ramdom 5 text image and labels predict
    random_indexes = random.sample(range(x_test.shape[0]),5)
    #Visualizing the orginal images
    fig, axes = plt.subplots(nrows=1, ncols=5)
    original_images = [x_test[i].reshape((28, 28)) for i in random_indexes]
    i = 0
    for ax in axes:
        ax.imshow(original_images[i], cmap ='gist_gray')
        i += 1
    fig.tight_layout()

    print([labels_text[i] for i in y_hat[random_indexes]])

