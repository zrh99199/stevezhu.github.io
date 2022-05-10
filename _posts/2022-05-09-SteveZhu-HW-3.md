---
layout: post
title: Homework 3
---
## Image Classification

In this blog, I will use different methods to build tensorflow machine learning models to classify the species of the animal in the images(dog/cat).

### 1. Load packages and obtain data

Import modules
```python
import os
from tensorflow.keras import utils
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
```

Load the data and save it as dataset
```python
# location of data
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

# download the data and extract it
path_to_zip = utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)

# construct paths
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

# parameters for datasets
BATCH_SIZE = 32
IMG_SIZE = (160, 160)

# construct train and validation datasets 
train_dataset = utils.image_dataset_from_directory(train_dir,
                                                   shuffle=True,
                                                   batch_size=BATCH_SIZE,
                                                   image_size=IMG_SIZE)

validation_dataset = utils.image_dataset_from_directory(validation_dir,
                                                        shuffle=True,
                                                        batch_size=BATCH_SIZE,
                                                        image_size=IMG_SIZE)

# construct the test dataset by taking every 5th observation out of the validation dataset
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)

# use autotune to read the data more efficiently
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
```

#### Working with datasets

I build a simple function to visualize what is in the dataset. The output of the function will be two rows of pictures, the first row is the randomly picked cat pictures and the second row is the randomly picked dog pictures.

``` python
def vis_fun(ds):
  for images, labels in ds:
    # find the cat and dog index in the dataset and randomly choose three for each specie
    # label is 0 for cats and 1 for dogs
    cat_index = np.where(labels == 0)[0]
    dog_index = np.where(labels == 1)[0]
    cat_choice = np.random.choice(cat_index, size = 3, replace = False)
    dog_choice = np.random.choice(dog_index, size = 3, replace = False)

    plt.figure(figsize = (10,10))
    for i in range(3):
      plt.subplot(2,3,i+1)
      plt.imshow(images[cat_choice[i]].numpy().astype("uint8"))
      plt.axis("off")
      
      plt.subplot(2,3,i+4)
      plt.imshow(images[dog_choice[i]].numpy().astype("uint8"))
      plt.axis("off")
      
vis_fun(train_dataset.take(1))
```
![HW3-plot1.png](/images/HW3-plot1.png)

#### Check label frequencies

```python
labels_iterator= train_dataset.unbatch().map(lambda image, label: label).as_numpy_iterator()
labels_list = list(labels_iterator)
```


```python
# the number of dog pictures
sum(labels_list)
```




    1000




```python
# the number of cat pictures
len(labels_list) - sum(labels_list)
```




    1000





Thus, the number of dog pictures and cat pictures are the same, so both species will get the same weight in the baseline machine learning model.


### 2. Model Building

#### First Model

```python
model1 = keras.models.Sequential([
      keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(160, 160, 3)),
      keras.layers.MaxPooling2D((2, 2)),
      keras.layers.Conv2D(32, (3, 3), activation='relu'),
      keras.layers.MaxPooling2D((2, 2)),
      keras.layers.Conv2D(64, (3, 3), activation='relu'), # n "pixels" x n "pixels" x 64
      keras.layers.Flatten(), # n^2 * 64 length vector
      keras.layers.Dropout(rate = 0.2), # drop 20%
      keras.layers.Dense(48, activation='relu'),
      keras.layers.Dense(32, activation='relu'),
      keras.layers.Dense(2) # number of classes(dog/cat)
])

model1.compile(optimizer='adam', 
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics = ['accuracy'])

history = model1.fit(train_dataset, 
                     epochs=20, 
                     validation_data=validation_dataset)
```

    Epoch 1/20
    63/63 [==============================] - 10s 161ms/step - loss: 13.7230 - accuracy: 0.5535 - val_loss: 0.6877 - val_accuracy: 0.5842
    Epoch 2/20
    63/63 [==============================] - 10s 156ms/step - loss: 0.6241 - accuracy: 0.6580 - val_loss: 0.7299 - val_accuracy: 0.5408
    Epoch 3/20
    63/63 [==============================] - 10s 163ms/step - loss: 0.5647 - accuracy: 0.7215 - val_loss: 0.8074 - val_accuracy: 0.5804
    Epoch 4/20
    63/63 [==============================] - 10s 165ms/step - loss: 0.4421 - accuracy: 0.7890 - val_loss: 0.9317 - val_accuracy: 0.5223
    Epoch 5/20
    63/63 [==============================] - 10s 163ms/step - loss: 0.3185 - accuracy: 0.8545 - val_loss: 1.2410 - val_accuracy: 0.5223
    Epoch 6/20
    63/63 [==============================] - 10s 159ms/step - loss: 0.3034 - accuracy: 0.8740 - val_loss: 1.3670 - val_accuracy: 0.5619
    Epoch 7/20
    63/63 [==============================] - 10s 162ms/step - loss: 0.1897 - accuracy: 0.9200 - val_loss: 1.5023 - val_accuracy: 0.5644
    Epoch 8/20
    63/63 [==============================] - 10s 160ms/step - loss: 0.1222 - accuracy: 0.9495 - val_loss: 1.9209 - val_accuracy: 0.5557
    Epoch 9/20
    63/63 [==============================] - 10s 160ms/step - loss: 0.0808 - accuracy: 0.9730 - val_loss: 2.2587 - val_accuracy: 0.5606
    Epoch 10/20
    63/63 [==============================] - 10s 163ms/step - loss: 0.1073 - accuracy: 0.9690 - val_loss: 1.7156 - val_accuracy: 0.5644
    Epoch 11/20
    63/63 [==============================] - 10s 161ms/step - loss: 0.1024 - accuracy: 0.9675 - val_loss: 2.1519 - val_accuracy: 0.5705
    Epoch 12/20
    63/63 [==============================] - 10s 155ms/step - loss: 0.0939 - accuracy: 0.9740 - val_loss: 2.2723 - val_accuracy: 0.5668
    Epoch 13/20
    63/63 [==============================] - 10s 157ms/step - loss: 0.0807 - accuracy: 0.9775 - val_loss: 2.6500 - val_accuracy: 0.5606
    Epoch 14/20
    63/63 [==============================] - 10s 157ms/step - loss: 0.0860 - accuracy: 0.9780 - val_loss: 2.2712 - val_accuracy: 0.5582
    Epoch 15/20
    63/63 [==============================] - 10s 156ms/step - loss: 0.0251 - accuracy: 0.9925 - val_loss: 2.9245 - val_accuracy: 0.5730
    Epoch 16/20
    63/63 [==============================] - 10s 157ms/step - loss: 0.0502 - accuracy: 0.9900 - val_loss: 3.2145 - val_accuracy: 0.5557
    Epoch 17/20
    63/63 [==============================] - 10s 156ms/step - loss: 0.0616 - accuracy: 0.9800 - val_loss: 2.0689 - val_accuracy: 0.5767
    Epoch 18/20
    63/63 [==============================] - 10s 157ms/step - loss: 0.0482 - accuracy: 0.9840 - val_loss: 2.9948 - val_accuracy: 0.5594
    Epoch 19/20
    63/63 [==============================] - 10s 157ms/step - loss: 0.0307 - accuracy: 0.9900 - val_loss: 3.3258 - val_accuracy: 0.5507
    Epoch 20/20
    63/63 [==============================] - 10s 159ms/step - loss: 0.0305 - accuracy: 0.9925 - val_loss: 2.8945 - val_accuracy: 0.5644
    


The accuracy of my model stabilized **between 95% and 99% ** during training and **between 55% and 57%** during validation.
Compared to the baseline model, I did slightly better than it.
This model experiences serious overfitting issue because the validation accuracy is much lower than the training accuracy.
#### Model with data augmentation

**RandomFlip Layer**

The RandomFlip Layer will randomly flip the picture by 90 degree
```python
randomflip = keras.Sequential([keras.layers.RandomFlip('horizontal')])

for images, labels in train_dataset.take(1):
    plt.figure(figsize = (10,10))
    for i in range(3):

        plt.subplot(3,4,4*i+1)
        im = images[i]

        plt.imshow(im.numpy().astype("uint8"))
        plt.axis("off")
        plt.title("original")
        
        for j in range(3):
            rf_im = randomflip(tf.expand_dims(im, 0),training=True)
            plt.subplot(3,4,4*i+2+j)
            plt.imshow(rf_im[0].numpy().astype("uint8"))
            plt.axis("off")
            plt.title("random flipped")
```
![HW3-plot2.png](/images/HW3-plot2.png)

**RandomRotation Layer**

The RandomFlip Layer will randomly flip the picture by the degree we choose
```python
randomrot = keras.layers.RandomRotation(0.2) # rotate by 20 degree

for images, labels in train_dataset.take(1):
    rf_im = [0, 0, 0]
    plt.figure(figsize = (10,10))
    for i in range(3):

        plt.subplot(3,4,4*i+1)
        im = images[i]

        plt.imshow(im.numpy().astype("uint8"))
        plt.axis("off")
        plt.title("original")
        
        for j in range(3):
            rf_im = randomrot(tf.expand_dims(im, 0),training=True)
            plt.subplot(3,4,4*i+2+j)
            plt.imshow(rf_im[0].numpy().astype("uint8"))
            plt.axis("off")
            plt.title("random flipped")
```
![HW3-plot3.png](/images/HW3-plot3.png)

```python
model2 = keras.models.Sequential([
      keras.layers.RandomFlip(input_shape=(160, 160, 3)),                          
      keras.layers.RandomRotation(0.2),     
      keras.layers.Conv2D(32, (3, 3), activation='relu'),
      keras.layers.MaxPooling2D((2, 2)),
      keras.layers.Conv2D(32, (3, 3), activation='relu'),
      keras.layers.MaxPooling2D((2, 2)),
      keras.layers.Conv2D(64, (3, 3), activation='relu'), # n "pixels" x n "pixels" x 64
      keras.layers.Flatten(), # n^2 * 64 length vector
      keras.layers.Dropout(rate = 0.2), # drop 20%
      keras.layers.Dense(64, activation='relu'),
      keras.layers.Dense(32, activation='relu'),
      keras.layers.Dense(2) # number of classes(dog/cat)
])

model2.compile(optimizer='adam', 
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics = ['accuracy'])

history = model2.fit(train_dataset, 
                     epochs=20, 
                     validation_data=validation_dataset)
```

    Epoch 1/20
    63/63 [==============================] - 11s 172ms/step - loss: 19.1171 - accuracy: 0.5280 - val_loss: 0.6777 - val_accuracy: 0.5743
    Epoch 2/20
    63/63 [==============================] - 11s 175ms/step - loss: 0.6760 - accuracy: 0.5750 - val_loss: 0.6429 - val_accuracy: 0.6225
    Epoch 3/20
    63/63 [==============================] - 11s 171ms/step - loss: 0.6748 - accuracy: 0.5855 - val_loss: 0.6611 - val_accuracy: 0.6139
    Epoch 4/20
    63/63 [==============================] - 11s 167ms/step - loss: 0.6898 - accuracy: 0.5380 - val_loss: 0.6787 - val_accuracy: 0.6002
    Epoch 5/20
    63/63 [==============================] - 11s 170ms/step - loss: 0.6769 - accuracy: 0.5900 - val_loss: 0.6550 - val_accuracy: 0.6126
    Epoch 6/20
    63/63 [==============================] - 11s 171ms/step - loss: 0.6742 - accuracy: 0.5885 - val_loss: 0.6599 - val_accuracy: 0.6324
    Epoch 7/20
    63/63 [==============================] - 11s 169ms/step - loss: 0.6652 - accuracy: 0.6075 - val_loss: 0.6673 - val_accuracy: 0.6089
    Epoch 8/20
    63/63 [==============================] - 11s 169ms/step - loss: 0.6722 - accuracy: 0.5800 - val_loss: 0.6918 - val_accuracy: 0.5210
    Epoch 9/20
    63/63 [==============================] - 11s 169ms/step - loss: 0.6841 - accuracy: 0.5680 - val_loss: 0.6763 - val_accuracy: 0.6040
    Epoch 10/20
    63/63 [==============================] - 11s 167ms/step - loss: 0.6736 - accuracy: 0.5855 - val_loss: 0.6722 - val_accuracy: 0.6077
    Epoch 11/20
    63/63 [==============================] - 11s 169ms/step - loss: 0.6747 - accuracy: 0.5955 - val_loss: 0.6801 - val_accuracy: 0.5804
    Epoch 12/20
    63/63 [==============================] - 11s 177ms/step - loss: 0.6644 - accuracy: 0.6120 - val_loss: 0.6597 - val_accuracy: 0.6213
    Epoch 13/20
    63/63 [==============================] - 11s 176ms/step - loss: 0.6573 - accuracy: 0.6155 - val_loss: 0.6660 - val_accuracy: 0.6275
    Epoch 14/20
    63/63 [==============================] - 11s 181ms/step - loss: 0.6546 - accuracy: 0.6200 - val_loss: 0.6676 - val_accuracy: 0.5879
    Epoch 15/20
    63/63 [==============================] - 11s 174ms/step - loss: 0.6631 - accuracy: 0.5990 - val_loss: 0.6836 - val_accuracy: 0.5953
    Epoch 16/20
    63/63 [==============================] - 11s 170ms/step - loss: 0.6515 - accuracy: 0.6170 - val_loss: 0.6978 - val_accuracy: 0.5953
    Epoch 17/20
    63/63 [==============================] - 11s 171ms/step - loss: 0.6687 - accuracy: 0.5885 - val_loss: 0.6692 - val_accuracy: 0.5854
    Epoch 18/20
    63/63 [==============================] - 11s 171ms/step - loss: 0.6595 - accuracy: 0.6215 - val_loss: 0.6567 - val_accuracy: 0.6015
    Epoch 19/20
    63/63 [==============================] - 11s 173ms/step - loss: 0.6590 - accuracy: 0.5975 - val_loss: 0.6987 - val_accuracy: 0.5260
    Epoch 20/20
    63/63 [==============================] - 11s 170ms/step - loss: 0.6496 - accuracy: 0.6275 - val_loss: 0.6640 - val_accuracy: 0.6002
    


The accuracy of my model stabilized **between 58% and 62%** during training and **between 58% and 62%** during validation.
Compared to the baseline model, I did slightly better than it.
This model doesn't have serious overfitting issue because the validation accuracy is almost the same as the trainning accuracy.
#### Model with data preprocessing
```python
i = tf.keras.Input(shape=(160, 160, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(i)
preprocessor = tf.keras.Model(inputs = [i], outputs = [x])

model3 = keras.models.Sequential([
      preprocessor,
      keras.layers.RandomFlip(),                          
      keras.layers.RandomRotation(0.2),                          
      keras.layers.Conv2D(32, (3, 3), activation='relu'),
      keras.layers.MaxPooling2D((2, 2)),
      keras.layers.Conv2D(32, (3, 3), activation='relu'),
      keras.layers.MaxPooling2D((2, 2)),
      keras.layers.Conv2D(64, (3, 3), activation='relu'), # n "pixels" x n "pixels" x 64
      keras.layers.Flatten(), # n^2 * 64 length vector
      keras.layers.Dropout(rate = 0.2), # drop 20%
      keras.layers.Dense(64, activation='relu'),
      keras.layers.Dense(32, activation='relu'),
      keras.layers.Dense(2) # number of classes(dog/cat)
])

model3.compile(optimizer='adam', 
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics = ['accuracy'])

history = model3.fit(train_dataset, 
                     epochs=20, 
                     validation_data=validation_dataset)
```

    Epoch 1/20
    63/63 [==============================] - 11s 171ms/step - loss: 0.7011 - accuracy: 0.5120 - val_loss: 0.6434 - val_accuracy: 0.6200
    Epoch 2/20
    63/63 [==============================] - 11s 170ms/step - loss: 0.6540 - accuracy: 0.5995 - val_loss: 0.6516 - val_accuracy: 0.5730
    Epoch 3/20
    63/63 [==============================] - 11s 173ms/step - loss: 0.6423 - accuracy: 0.6075 - val_loss: 0.6447 - val_accuracy: 0.6089
    Epoch 4/20
    63/63 [==============================] - 11s 173ms/step - loss: 0.6360 - accuracy: 0.6240 - val_loss: 0.6214 - val_accuracy: 0.6498
    Epoch 5/20
    63/63 [==============================] - 11s 174ms/step - loss: 0.6141 - accuracy: 0.6470 - val_loss: 0.6116 - val_accuracy: 0.6757
    Epoch 6/20
    63/63 [==============================] - 11s 171ms/step - loss: 0.6035 - accuracy: 0.6695 - val_loss: 0.5785 - val_accuracy: 0.6733
    Epoch 7/20
    63/63 [==============================] - 11s 172ms/step - loss: 0.5798 - accuracy: 0.6850 - val_loss: 0.6048 - val_accuracy: 0.6696
    Epoch 8/20
    63/63 [==============================] - 11s 173ms/step - loss: 0.5913 - accuracy: 0.6735 - val_loss: 0.6289 - val_accuracy: 0.6324
    Epoch 9/20
    63/63 [==============================] - 11s 170ms/step - loss: 0.5922 - accuracy: 0.6755 - val_loss: 0.5814 - val_accuracy: 0.7054
    Epoch 10/20
    63/63 [==============================] - 11s 171ms/step - loss: 0.5575 - accuracy: 0.7005 - val_loss: 0.5497 - val_accuracy: 0.7203
    Epoch 11/20
    63/63 [==============================] - 11s 171ms/step - loss: 0.5577 - accuracy: 0.7130 - val_loss: 0.5916 - val_accuracy: 0.6993
    Epoch 12/20
    63/63 [==============================] - 11s 171ms/step - loss: 0.5532 - accuracy: 0.7180 - val_loss: 0.5504 - val_accuracy: 0.7252
    Epoch 13/20
    63/63 [==============================] - 11s 171ms/step - loss: 0.5384 - accuracy: 0.7225 - val_loss: 0.5426 - val_accuracy: 0.7215
    Epoch 14/20
    63/63 [==============================] - 11s 173ms/step - loss: 0.5261 - accuracy: 0.7350 - val_loss: 0.5352 - val_accuracy: 0.7450
    Epoch 15/20
    63/63 [==============================] - 11s 171ms/step - loss: 0.5292 - accuracy: 0.7325 - val_loss: 0.5540 - val_accuracy: 0.7141
    Epoch 16/20
    63/63 [==============================] - 11s 173ms/step - loss: 0.5258 - accuracy: 0.7270 - val_loss: 0.5637 - val_accuracy: 0.7265
    Epoch 17/20
    63/63 [==============================] - 11s 173ms/step - loss: 0.5269 - accuracy: 0.7395 - val_loss: 0.5351 - val_accuracy: 0.7327
    Epoch 18/20
    63/63 [==============================] - 11s 173ms/step - loss: 0.5136 - accuracy: 0.7475 - val_loss: 0.5482 - val_accuracy: 0.7277
    Epoch 19/20
    63/63 [==============================] - 11s 175ms/step - loss: 0.5010 - accuracy: 0.7510 - val_loss: 0.5562 - val_accuracy: 0.7252
    Epoch 20/20
    63/63 [==============================] - 11s 172ms/step - loss: 0.5155 - accuracy: 0.7435 - val_loss: 0.5763 - val_accuracy: 0.7054
    


The accuracy of my model stabilized **between 70% and 75%** during training and **between 70% and 74%** during validation.
Compared to the baseline model, I did much better than it.
This model doesn't have serious overfitting issue because the validation accuracy is almost the same as the trainning accuracy.

#### Model with transfer learning
```python
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

i = tf.keras.Input(shape=IMG_SHAPE)
x = base_model(i, training = False)
base_model_layer = tf.keras.Model(inputs = [i], outputs = [x])

model4 = keras.models.Sequential([
      preprocessor,
      keras.layers.RandomFlip(),                          
      keras.layers.RandomRotation(0.2),
      base_model_layer,
      keras.layers.Flatten(), # n^2 * 64 length vector
      keras.layers.Dense(2) # number of classes in your dataset
])

model4.compile(optimizer='adam', 
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics = ['accuracy'])

history = model4.fit(train_dataset, 
                     epochs=20, 
                     validation_data=validation_dataset)
```

    Epoch 1/20
    63/63 [==============================] - 12s 177ms/step - loss: 1.2558 - accuracy: 0.8450 - val_loss: 0.2748 - val_accuracy: 0.9629
    Epoch 2/20
    63/63 [==============================] - 11s 172ms/step - loss: 0.4705 - accuracy: 0.9205 - val_loss: 0.3208 - val_accuracy: 0.9554
    Epoch 3/20
    63/63 [==============================] - 11s 173ms/step - loss: 0.5691 - accuracy: 0.9085 - val_loss: 0.3323 - val_accuracy: 0.9505
    Epoch 4/20
    63/63 [==============================] - 11s 172ms/step - loss: 0.5053 - accuracy: 0.9215 - val_loss: 0.3576 - val_accuracy: 0.9592
    Epoch 5/20
    63/63 [==============================] - 11s 172ms/step - loss: 0.6638 - accuracy: 0.9240 - val_loss: 0.8721 - val_accuracy: 0.9084
    Epoch 6/20
    63/63 [==============================] - 11s 174ms/step - loss: 0.6721 - accuracy: 0.9230 - val_loss: 0.2936 - val_accuracy: 0.9604
    Epoch 7/20
    63/63 [==============================] - 11s 173ms/step - loss: 0.4724 - accuracy: 0.9390 - val_loss: 0.5092 - val_accuracy: 0.9493
    Epoch 8/20
    63/63 [==============================] - 11s 173ms/step - loss: 0.3916 - accuracy: 0.9440 - val_loss: 0.3350 - val_accuracy: 0.9653
    Epoch 9/20
    63/63 [==============================] - 11s 175ms/step - loss: 0.5852 - accuracy: 0.9435 - val_loss: 0.3870 - val_accuracy: 0.9629
    Epoch 10/20
    63/63 [==============================] - 11s 172ms/step - loss: 0.4853 - accuracy: 0.9365 - val_loss: 0.5878 - val_accuracy: 0.9468
    Epoch 11/20
    63/63 [==============================] - 11s 173ms/step - loss: 0.4744 - accuracy: 0.9525 - val_loss: 0.4063 - val_accuracy: 0.9542
    Epoch 12/20
    63/63 [==============================] - 11s 174ms/step - loss: 0.5270 - accuracy: 0.9405 - val_loss: 0.3614 - val_accuracy: 0.9653
    Epoch 13/20
    63/63 [==============================] - 11s 174ms/step - loss: 0.5501 - accuracy: 0.9435 - val_loss: 0.3191 - val_accuracy: 0.9579
    Epoch 14/20
    63/63 [==============================] - 11s 175ms/step - loss: 0.3697 - accuracy: 0.9555 - val_loss: 0.4342 - val_accuracy: 0.9579
    Epoch 15/20
    63/63 [==============================] - 11s 174ms/step - loss: 0.4216 - accuracy: 0.9450 - val_loss: 0.3545 - val_accuracy: 0.9616
    Epoch 16/20
    63/63 [==============================] - 11s 174ms/step - loss: 0.5178 - accuracy: 0.9490 - val_loss: 0.4330 - val_accuracy: 0.9616
    Epoch 17/20
    63/63 [==============================] - 11s 174ms/step - loss: 0.4064 - accuracy: 0.9510 - val_loss: 0.4086 - val_accuracy: 0.9616
    Epoch 18/20
    63/63 [==============================] - 11s 174ms/step - loss: 0.4309 - accuracy: 0.9555 - val_loss: 0.4209 - val_accuracy: 0.9728
    Epoch 19/20
    63/63 [==============================] - 11s 175ms/step - loss: 0.4516 - accuracy: 0.9565 - val_loss: 0.4693 - val_accuracy: 0.9629
    Epoch 20/20
    63/63 [==============================] - 11s 174ms/step - loss: 0.3138 - accuracy: 0.9620 - val_loss: 0.5924 - val_accuracy: 0.9567
    


The accuracy of my model stabilized **between 94% and 96%** during training and **between 95% and 96%** during validation.
Compared to the baseline model, I did much much better than it.
This model doesn't have serious overfitting issue because the validation accuracy is almost the same as the trainning accuracy

### 3. Score on Test Data

```python
model1.evaluate(test_dataset, verbose=1)
model2.evaluate(test_dataset, verbose=1)
model3.evaluate(test_dataset, verbose=1)
model4.evaluate(test_dataset, verbose=1)
```

    6/6 [==============================] - 0s 32ms/step - loss: 3.2152 - accuracy: 0.6302
    6/6 [==============================] - 0s 34ms/step - loss: 0.6592 - accuracy: 0.6823
    6/6 [==============================] - 0s 35ms/step - loss: 0.5746 - accuracy: 0.7188
    6/6 [==============================] - 1s 114ms/step - loss: 0.5372 - accuracy: 0.9688
    
The best model is the last one with the transfer learning. It makes sense because that model is using other well-trained model to train our model. For the first three models, the accuracy has improved since I add more specific layers to my model.
