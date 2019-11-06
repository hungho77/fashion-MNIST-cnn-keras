# Fashion-MNIST classification using Convolution Neural Network with Keras

## Dataset
### Fashion-MNIST
Link: [Fashion-MNIST](https://www.kaggle.com/zalando-research/fashionmnist)

## Software requirements
* Python 3.6, TensorFlow 1.11.0, Keras 2.2.4, numpy, matplotlib, scikit-learn, h5py

## Training
* Open file fashion-mnist-cnn.ipynb and run on colab
* or run file fashion-mnist-cnn.py with command:

python fashion-mnist-cnn.py --path [str] --epochs [int] --batch_size [int]


## CNN Model 
The network topology can be summarized as follows:
- Batch normalization layer
- Convolutional layer with 64 feature maps of size 3×3.
- Convolutional layer with 64 feature maps of size 3×3.
- Pooling layer taking the max over 2*2 patches.
- Dropout layer with a probability of 10%.
- Convolutional layer with 32 feature maps of size 3×3.
- Convolutional layer with 32 feature maps of size 3×3.
- Pooling layer taking the max over 2*2 patches.
- Dropout layer with a probability of 30%.
- Flatten layer.
- Fully connected layer with 256 node and rectifier activation.
- Dropout layer with a probability of 50%.
- Fully connected layer with 64 node and rectifier activation.
- Batch normalization layer
- Output layer with 10 node with softmax.

## Results
Model traing on training set 60,000 samples which devide into 48,000 training set and 12,000 valid set.
I evaluate on Test set with 10,000 samples.

* Traning accuracy: 99.27%

* Validation accuracy: 93.51%

* Test accuracy: 94.25%

The following picture shows the trend of the Accuracy of the final learning:
![alt text](https://github.com/hohung77/fashion-MNIST-cnn-keras/blob/master/model%20loss.png)
![alt text](https://github.com/hohung77/fashion-MNIST-cnn-keras/blob/master/model%20accuracy.png)

## Experiment
Accuracy

| Model | Parametter | train_acc | val_acc |
| --- | --- | --- | --- |
| 1Conv + 1Pooling + 2FC + Softmax | 607,818  | 97.92% | 91.42% |
| 2Conv + Pooling + 2FC + Softmax | 644,746  | 99.67% | 92.83% |
| 2Conv + 2Pooling + 2FC + Sofmax | 857,738 | 99.93% | 92.68% |
| 3Conv + 2Pooling + 2FC + Softmax | 894,666 | 99.80% | 93.29% |
| 3Conv + 2batch_normalization + 2Pooling + 2FC + Softmax | 894,926 | 99.97% | 93.46% |
| 4Conv + 2batch_normalization + 2Pooling + 2FC + Softmax | 484,172 | 99.27% | **93.51%** |

