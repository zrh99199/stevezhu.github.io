---
layout: post
title: Homework 4
---
## Fake News Analysis

In this blog, I will build three models to determine whether the news is fake

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

Load the data and prepossess the data
```python
sw = stopwords.words('english')

def make_dataset(df):
  """
  Input a dataframe, clean its title and text by removing stopwords
  Output a tensorflow dataset with 'title' and 'text' as features, 'fake' as label
  """

  #make a copy to make sure no change to the original data
  df_copy = df
  df_copy['title'] = df_copy['title'].apply(lambda x: ' '.join([word for word in x.split() if word not in (sw)]))
  df_copy['text'] = df_copy['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (sw)]))

  dataset = tf.data.Dataset.from_tensor_slices(
      (
        # dictionary for input data/features
        { "title": df_copy[["title"]],
          "text": df_copy['text']
        },
        # dictionary for output data/labels
        { "fake": df_copy[["fake"]]
        }
      )
  )

  dataset.batch(100)
  dataset = dataset.shuffle(buffer_size = len(dataset))

  return dataset
  
ds = make_dataset(df)
```

Split 20% of the data as validation data

```python
train_size = int(0.8*len(ds))
val_size   = int(0.2*len(ds))

train = ds.take(train_size).batch(20)
val = ds.skip(train_size).take(val_size).batch(20)
```

Take a look of the base rate

```python
fake_index = np.concatenate([y.get('fake') for x, y in ds], axis=0)

#base rate(sum(fake_index) refers to the number of fake nums, len(fake_index) refers to the number of total news)
sum(fake_index)/len(fake_index)
```




    0.522963160942581



Make TextVectorization for our two features(title and text)
```python
#preparing a text vectorization layer for tf model
size_vocabulary = 2000

def standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    no_punctuation = tf.strings.regex_replace(lowercase,
                                  '[%s]' % re.escape(string.punctuation),'')
    return no_punctuation 

title_vectorize_layer = TextVectorization(
    standardize=standardization,
    max_tokens=size_vocabulary, # only consider this many words
    output_mode='int',
    output_sequence_length=500) 

title_vectorize_layer.adapt(train.map(lambda x, y: x["title"]))

text_vectorize_layer = TextVectorization(
    standardize=standardization,
    max_tokens=size_vocabulary, # only consider this many words
    output_mode='int',
    output_sequence_length=500) 

text_vectorize_layer.adapt(train.map(lambda x, y: x["text"]))
```

### 2. Model Building

I will build three models via tensorflow functional API.

#### First Model
The first one would be a model only with the article title as input

```python
# model with only the title as output
title_input = keras.Input(
    shape=(1,),
    name = "title", # same name as the dictionary key in the dataset
    dtype = "string"
)
title_features = title_vectorize_layer(title_input) # apply this "function TextVectorization layer" to lyrics_input
title_features = layers.Embedding(size_vocabulary, output_dim = 10, name="embedding")(title_features)
title_features = layers.Dropout(0.2)(title_features)
title_features = layers.GlobalAveragePooling1D()(title_features)
title_features = layers.Dropout(0.2)(title_features)
title_features = layers.Dense(32, activation='relu')(title_features)
title_output = layers.Dense(2, name="fake")(title_features) 
model1 = keras.Model(
    inputs = title_input,
    outputs = title_output
)
model1.summary()
```
```

    Model: "model"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     title (InputLayer)          [(None, 1)]               0         
                                                                     
     text_vectorization (TextVec  (None, 500)              0         
     torization)                                                     
                                                                     
     embedding (Embedding)       (None, 500, 10)           20000     
                                                                     
     dropout (Dropout)           (None, 500, 10)           0         
                                                                     
     global_average_pooling1d (G  (None, 10)               0         
     lobalAveragePooling1D)                                          
                                                                     
     dropout_1 (Dropout)         (None, 10)                0         
                                                                     
     dense (Dense)               (None, 32)                352       
                                                                     
     fake (Dense)                (None, 2)                 66        
                                                                     
    =================================================================
    Total params: 20,418
    Trainable params: 20,418
    Non-trainable params: 0
    _________________________________________________________________
    

```python
model1.compile(optimizer="adam",
              loss = losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
```


```python
history1 = model1.fit(train, 
                    validation_data=val,
                    epochs = 20)
```

    Epoch 1/20
    898/898 [==============================] - 6s 6ms/step - loss: 0.6128 - accuracy: 0.6730 - val_loss: 0.3270 - val_accuracy: 0.9076
    Epoch 2/20
    898/898 [==============================] - 10s 11ms/step - loss: 0.1770 - accuracy: 0.9488 - val_loss: 0.1058 - val_accuracy: 0.9668
    Epoch 3/20
    898/898 [==============================] - 11s 13ms/step - loss: 0.0985 - accuracy: 0.9679 - val_loss: 0.0781 - val_accuracy: 0.9788
    Epoch 4/20
    898/898 [==============================] - 6s 6ms/step - loss: 0.0753 - accuracy: 0.9748 - val_loss: 0.0603 - val_accuracy: 0.9795
    Epoch 5/20
    898/898 [==============================] - 6s 7ms/step - loss: 0.0661 - accuracy: 0.9781 - val_loss: 0.0521 - val_accuracy: 0.9804
    Epoch 6/20
    898/898 [==============================] - 9s 10ms/step - loss: 0.0612 - accuracy: 0.9792 - val_loss: 0.0589 - val_accuracy: 0.9793
    Epoch 7/20
    898/898 [==============================] - 6s 7ms/step - loss: 0.0567 - accuracy: 0.9806 - val_loss: 0.0444 - val_accuracy: 0.9846
    Epoch 8/20
    898/898 [==============================] - 6s 6ms/step - loss: 0.0546 - accuracy: 0.9806 - val_loss: 0.0411 - val_accuracy: 0.9835
    Epoch 9/20
    898/898 [==============================] - 7s 8ms/step - loss: 0.0495 - accuracy: 0.9830 - val_loss: 0.0345 - val_accuracy: 0.9893
    Epoch 10/20
    898/898 [==============================] - 7s 7ms/step - loss: 0.0466 - accuracy: 0.9841 - val_loss: 0.0312 - val_accuracy: 0.9913
    Epoch 11/20
    898/898 [==============================] - 7s 8ms/step - loss: 0.0482 - accuracy: 0.9836 - val_loss: 0.0359 - val_accuracy: 0.9889
    Epoch 12/20
    898/898 [==============================] - 7s 7ms/step - loss: 0.0413 - accuracy: 0.9859 - val_loss: 0.0380 - val_accuracy: 0.9895
    Epoch 13/20
    898/898 [==============================] - 8s 9ms/step - loss: 0.0418 - accuracy: 0.9849 - val_loss: 0.0364 - val_accuracy: 0.9869
    Epoch 14/20
    898/898 [==============================] - 7s 7ms/step - loss: 0.0415 - accuracy: 0.9863 - val_loss: 0.0358 - val_accuracy: 0.9884
    Epoch 15/20
    898/898 [==============================] - 6s 7ms/step - loss: 0.0398 - accuracy: 0.9870 - val_loss: 0.0354 - val_accuracy: 0.9864
    Epoch 16/20
    898/898 [==============================] - 6s 7ms/step - loss: 0.0390 - accuracy: 0.9876 - val_loss: 0.0294 - val_accuracy: 0.9902
    Epoch 17/20
    898/898 [==============================] - 6s 7ms/step - loss: 0.0379 - accuracy: 0.9865 - val_loss: 0.0488 - val_accuracy: 0.9822
    Epoch 18/20
    898/898 [==============================] - 6s 7ms/step - loss: 0.0387 - accuracy: 0.9861 - val_loss: 0.0455 - val_accuracy: 0.9833
    Epoch 19/20
    898/898 [==============================] - 7s 8ms/step - loss: 0.0380 - accuracy: 0.9872 - val_loss: 0.0297 - val_accuracy: 0.9900
    Epoch 20/20
    898/898 [==============================] - 7s 7ms/step - loss: 0.0340 - accuracy: 0.9889 - val_loss: 0.0220 - val_accuracy: 0.9947
    

Visualize the accuracy and validation accuracy
```python
plt.plot(history1.history["accuracy"])
plt.plot(history1.history["val_accuracy"])
```
![HW4-plot1.png](/images/HW4-plot1.png)

#### Second Model
The first one would be a model only with the article text as input

```python
# model with only the text as output
text_input = keras.Input(
    shape=(1,),
    name = "text", # same name as the dictionary key in the dataset
    dtype = "string"
)
text_features = text_vectorize_layer(text_input) # apply this "function TextVectorization layer" to lyrics_input
text_features = layers.Embedding(size_vocabulary, output_dim = 10, name="embedding2")(text_features)
text_features = layers.Dropout(0.2)(text_features)
text_features = layers.GlobalAveragePooling1D()(text_features)
text_features = layers.Dropout(0.2)(text_features)
text_features = layers.Dense(32, activation='relu')(text_features)
text_output = layers.Dense(2, name="fake")(text_features) 
model2 = keras.Model(
    inputs = text_input,
    outputs = text_output
)
model2.summary()
```

    Model: "model_1"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     text (InputLayer)           [(None, 1)]               0         
                                                                     
     text_vectorization_1 (TextV  (None, 500)              0         
     ectorization)                                                   
                                                                     
     embedding2 (Embedding)      (None, 500, 10)           20000     
                                                                     
     dropout_2 (Dropout)         (None, 500, 10)           0         
                                                                     
     global_average_pooling1d_1   (None, 10)               0         
     (GlobalAveragePooling1D)                                        
                                                                     
     dropout_3 (Dropout)         (None, 10)                0         
                                                                     
     dense_1 (Dense)             (None, 32)                352       
                                                                     
     fake (Dense)                (None, 2)                 66        
                                                                     
    =================================================================
    Total params: 20,418
    Trainable params: 20,418
    Non-trainable params: 0
    _________________________________________________________________
    


```python
model2.compile(optimizer="adam",
              loss = losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
```


```python
history2 = model2.fit(train, 
                    validation_data=val,
                    epochs = 20)
```

    Epoch 1/20
    898/898 [==============================] - 12s 12ms/step - loss: 0.3467 - accuracy: 0.8575 - val_loss: 0.1245 - val_accuracy: 0.9677
    Epoch 2/20
    898/898 [==============================] - 9s 10ms/step - loss: 0.1149 - accuracy: 0.9679 - val_loss: 0.0782 - val_accuracy: 0.9797
    Epoch 3/20
    898/898 [==============================] - 10s 11ms/step - loss: 0.0852 - accuracy: 0.9769 - val_loss: 0.0578 - val_accuracy: 0.9869
    Epoch 4/20
    898/898 [==============================] - 9s 10ms/step - loss: 0.0688 - accuracy: 0.9804 - val_loss: 0.0585 - val_accuracy: 0.9857
    Epoch 5/20
    898/898 [==============================] - 7s 8ms/step - loss: 0.0597 - accuracy: 0.9834 - val_loss: 0.0483 - val_accuracy: 0.9898
    Epoch 6/20
    898/898 [==============================] - 9s 10ms/step - loss: 0.0515 - accuracy: 0.9852 - val_loss: 0.0350 - val_accuracy: 0.9931
    Epoch 7/20
    898/898 [==============================] - 12s 14ms/step - loss: 0.0441 - accuracy: 0.9872 - val_loss: 0.0310 - val_accuracy: 0.9924
    Epoch 8/20
    898/898 [==============================] - 9s 9ms/step - loss: 0.0403 - accuracy: 0.9887 - val_loss: 0.0255 - val_accuracy: 0.9951
    Epoch 9/20
    898/898 [==============================] - 10s 11ms/step - loss: 0.0356 - accuracy: 0.9898 - val_loss: 0.0239 - val_accuracy: 0.9944
    Epoch 10/20
    898/898 [==============================] - 8s 9ms/step - loss: 0.0284 - accuracy: 0.9916 - val_loss: 0.0163 - val_accuracy: 0.9962
    Epoch 11/20
    898/898 [==============================] - 9s 10ms/step - loss: 0.0291 - accuracy: 0.9918 - val_loss: 0.0154 - val_accuracy: 0.9969
    Epoch 12/20
    898/898 [==============================] - 12s 13ms/step - loss: 0.0268 - accuracy: 0.9925 - val_loss: 0.0232 - val_accuracy: 0.9940
    Epoch 13/20
    898/898 [==============================] - 9s 10ms/step - loss: 0.0242 - accuracy: 0.9933 - val_loss: 0.0216 - val_accuracy: 0.9967
    Epoch 14/20
    898/898 [==============================] - 7s 8ms/step - loss: 0.0229 - accuracy: 0.9930 - val_loss: 0.0138 - val_accuracy: 0.9980
    Epoch 15/20
    898/898 [==============================] - 8s 9ms/step - loss: 0.0218 - accuracy: 0.9927 - val_loss: 0.0168 - val_accuracy: 0.9960
    Epoch 16/20
    898/898 [==============================] - 7s 8ms/step - loss: 0.0236 - accuracy: 0.9931 - val_loss: 0.0125 - val_accuracy: 0.9973
    Epoch 17/20
    898/898 [==============================] - 7s 8ms/step - loss: 0.0214 - accuracy: 0.9929 - val_loss: 0.0171 - val_accuracy: 0.9944
    Epoch 18/20
    898/898 [==============================] - 7s 8ms/step - loss: 0.0174 - accuracy: 0.9948 - val_loss: 0.0118 - val_accuracy: 0.9984
    Epoch 19/20
    898/898 [==============================] - 8s 9ms/step - loss: 0.0184 - accuracy: 0.9940 - val_loss: 0.0079 - val_accuracy: 0.9980
    Epoch 20/20
    898/898 [==============================] - 8s 9ms/step - loss: 0.0166 - accuracy: 0.9955 - val_loss: 0.0087 - val_accuracy: 0.9978
    

Visualize the accuracy and validation accuracy
```python
plt.plot(history2.history["accuracy"])
plt.plot(history2.history["val_accuracy"])
```
![HW4-plot2.png](/images/HW4-plot2.png)

#### Third Model
The third model will use both the article title and text as input
```python
merged_features = layers.concatenate([title_features, text_features], axis = 1)
merged_features = layers.Dense(32, activation='relu')(merged_features)
merged_output = layers.Dense(2, name="fake")(merged_features)
model3 = keras.Model(
    inputs = [title_input, text_input],
    outputs = merged_output
)
model3.summary()
```

    Model: "model_2"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to                     
    ==================================================================================================
     title (InputLayer)             [(None, 1)]          0           []                               
                                                                                                      
     text (InputLayer)              [(None, 1)]          0           []                               
                                                                                                      
     text_vectorization (TextVector  (None, 500)         0           ['title[0][0]']                  
     ization)                                                                                         
                                                                                                      
     text_vectorization_1 (TextVect  (None, 500)         0           ['text[0][0]']                   
     orization)                                                                                       
                                                                                                      
     embedding (Embedding)          (None, 500, 10)      20000       ['text_vectorization[0][0]']     
                                                                                                      
     embedding2 (Embedding)         (None, 500, 10)      20000       ['text_vectorization_1[0][0]']   
                                                                                                      
     dropout (Dropout)              (None, 500, 10)      0           ['embedding[0][0]']              
                                                                                                      
     dropout_2 (Dropout)            (None, 500, 10)      0           ['embedding2[0][0]']             
                                                                                                      
     global_average_pooling1d (Glob  (None, 10)          0           ['dropout[0][0]']                
     alAveragePooling1D)                                                                              
                                                                                                      
     global_average_pooling1d_1 (Gl  (None, 10)          0           ['dropout_2[0][0]']              
     obalAveragePooling1D)                                                                            
                                                                                                      
     dropout_1 (Dropout)            (None, 10)           0           ['global_average_pooling1d[0][0]'
                                                                     ]                                
                                                                                                      
     dropout_3 (Dropout)            (None, 10)           0           ['global_average_pooling1d_1[0][0
                                                                     ]']                              
                                                                                                      
     dense (Dense)                  (None, 32)           352         ['dropout_1[0][0]']              
                                                                                                      
     dense_1 (Dense)                (None, 32)           352         ['dropout_3[0][0]']              
                                                                                                      
     concatenate (Concatenate)      (None, 64)           0           ['dense[0][0]',                  
                                                                      'dense_1[0][0]']                
                                                                                                      
     dense_2 (Dense)                (None, 32)           2080        ['concatenate[0][0]']            
                                                                                                      
     fake (Dense)                   (None, 2)            66          ['dense_2[0][0]']                
                                                                                                      
    ==================================================================================================
    Total params: 42,850
    Trainable params: 42,850
    Non-trainable params: 0
    __________________________________________________________________________________________________
    


```python
model3.compile(optimizer="adam",
              loss = losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
```


```python
history3 = model3.fit(train, 
                    validation_data=val,
                    epochs = 20)
```

    Epoch 1/20
    898/898 [==============================] - 15s 15ms/step - loss: 0.0219 - accuracy: 0.9978 - val_loss: 0.0018 - val_accuracy: 0.9998
    Epoch 2/20
    898/898 [==============================] - 9s 10ms/step - loss: 0.0049 - accuracy: 0.9986 - val_loss: 0.0038 - val_accuracy: 0.9989
    Epoch 3/20
    898/898 [==============================] - 9s 10ms/step - loss: 0.0040 - accuracy: 0.9988 - val_loss: 0.0024 - val_accuracy: 0.9993
    Epoch 4/20
    898/898 [==============================] - 9s 10ms/step - loss: 0.0034 - accuracy: 0.9988 - val_loss: 5.5273e-04 - val_accuracy: 0.9998
    Epoch 5/20
    898/898 [==============================] - 10s 11ms/step - loss: 0.0038 - accuracy: 0.9988 - val_loss: 0.0014 - val_accuracy: 0.9993
    Epoch 6/20
    898/898 [==============================] - 13s 14ms/step - loss: 0.0033 - accuracy: 0.9988 - val_loss: 6.5620e-04 - val_accuracy: 0.9998
    Epoch 7/20
    898/898 [==============================] - 13s 14ms/step - loss: 0.0022 - accuracy: 0.9993 - val_loss: 6.9640e-04 - val_accuracy: 0.9998
    Epoch 8/20
    898/898 [==============================] - 9s 10ms/step - loss: 0.0041 - accuracy: 0.9987 - val_loss: 7.0335e-04 - val_accuracy: 0.9998
    Epoch 9/20
    898/898 [==============================] - 10s 11ms/step - loss: 0.0027 - accuracy: 0.9989 - val_loss: 0.0015 - val_accuracy: 0.9996
    Epoch 10/20
    898/898 [==============================] - 10s 11ms/step - loss: 0.0025 - accuracy: 0.9992 - val_loss: 1.5842e-04 - val_accuracy: 1.0000
    Epoch 11/20
    898/898 [==============================] - 9s 10ms/step - loss: 0.0011 - accuracy: 0.9996 - val_loss: 0.0152 - val_accuracy: 0.9949
    Epoch 12/20
    898/898 [==============================] - 9s 10ms/step - loss: 0.0038 - accuracy: 0.9988 - val_loss: 3.9243e-04 - val_accuracy: 0.9998
    Epoch 13/20
    898/898 [==============================] - 10s 11ms/step - loss: 0.0023 - accuracy: 0.9990 - val_loss: 3.9288e-04 - val_accuracy: 0.9998
    Epoch 14/20
    898/898 [==============================] - 8s 9ms/step - loss: 0.0018 - accuracy: 0.9997 - val_loss: 0.0011 - val_accuracy: 0.9998
    Epoch 15/20
    898/898 [==============================] - 12s 13ms/step - loss: 0.0018 - accuracy: 0.9994 - val_loss: 0.0010 - val_accuracy: 0.9991
    Epoch 16/20
    898/898 [==============================] - 11s 13ms/step - loss: 0.0031 - accuracy: 0.9992 - val_loss: 8.4163e-05 - val_accuracy: 1.0000
    Epoch 17/20
    898/898 [==============================] - 9s 10ms/step - loss: 0.0018 - accuracy: 0.9995 - val_loss: 3.1355e-04 - val_accuracy: 1.0000
    Epoch 18/20
    898/898 [==============================] - 10s 11ms/step - loss: 0.0014 - accuracy: 0.9996 - val_loss: 0.0012 - val_accuracy: 0.9996
    Epoch 19/20
    898/898 [==============================] - 9s 10ms/step - loss: 0.0042 - accuracy: 0.9990 - val_loss: 0.0047 - val_accuracy: 0.9982
    Epoch 20/20
    898/898 [==============================] - 9s 10ms/step - loss: 0.0017 - accuracy: 0.9994 - val_loss: 3.1226e-04 - val_accuracy: 0.9998
    

Visualize the accuracy and validation accuracy
```python
plt.plot(history3.history["accuracy"])
plt.plot(history3.history["val_accuracy"])
```
![HW4-plot3.png](/images/HW4-plot3.png)

As result, we see all three models reach a very high accuracy, the first one reaches 98%, the second and the third reach 99%.

### 3. Model Evaluation
Since all these three models obtain a very high accuracy rate in both training and validation data, let's see how they perform on the unseen test data.
```python
#Test models
test_url = "https://github.com/PhilChodrow/PIC16b/blob/master/datasets/fake_news_test.csv?raw=true"
test = pd.read_csv(test_url)
test = make_dataset(test)
test = test.batch(20)
```


```python
model1.evaluate(test, verbose=1)
```

    1123/1123 [==============================] - 4s 4ms/step - loss: 0.0658 - accuracy: 0.9819
    




    [0.06578266620635986, 0.9819145798683167]




```python
model2.evaluate(test, verbose=1)
```

    1123/1123 [==============================] - 7s 6ms/step - loss: 0.1432 - accuracy: 0.9743
    




    [0.14324592053890228, 0.9742527604103088]




```python
model3.evaluate(test, verbose=1)
```

    1123/1123 [==============================] - 8s 7ms/step - loss: 0.0212 - accuracy: 0.9953
    




    [0.021223340183496475, 0.9953227043151855]



As result, the model with both title and text as input obtains the highest accuracy rate (more than 99%). It makes sense because this model consider more related features in the training stage.

### 4. Visualize Embedding

Here, I build one 3D plot for title word embedding and one 2D plot for text word embedding.
```python
#Visualize word embedding in 'title' feature
title_weights = model3.get_layer('embedding').get_weights()[0] # get the weights from the embedding layer
title_vocab = title_vectorize_layer.get_vocabulary()                # get the vocabulary from our data prep for later

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
title_weights = pca.fit_transform(title_weights)

embedding_title = pd.DataFrame({
    'word' : title_vocab, 
    'x0'   : title_weights[:,0],
    'x1'   : title_weights[:,1],
    'x2'   : title_weights[:,2]
})

import plotly.express as px 
fig = px.scatter_3d(embedding_title, 
                 x = "x0", 
                 y = "x1", 
                 z = "x2",
                 size = [2]*len(embedding_title),
                 size_max = 10,
                 hover_name = "word",
                 )

fig.show()
```
{% include HW4-plotly1.html %}

```python
#Visualize word embedding in 'text' feature
text_weights = model3.get_layer('embedding').get_weights()[0] # get the weights from the embedding layer
text_vocab = text_vectorize_layer.get_vocabulary()                # get the vocabulary from our data prep for later

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
text_weights = pca.fit_transform(text_weights)

embedding_text = pd.DataFrame({
    'word' : text_vocab, 
    'x0'   : text_weights[:,0],
    'x1'   : text_weights[:,1]
})

import plotly.express as px 
fig = px.scatter(embedding_text, 
                 x = "x0", 
                 y = "x1", 
                 size = [2]*len(embedding_text),
                # size_max = 2,
                 hover_name = "word")

fig.show()
```
{% include HW4-plotly2.html %}

I found something interesting in my plots:
1. The word 'clinton's ' is very close to 'trump's' in the 3D plot.
2. The possessive adjectives are very close to each other(its, his, her) in the 3D plot.
3. In the 2D plot, 'trump' is near to 'white'.
4. In the 2D plot, 'kill' is close to 'winning'.
5. In the 2D plot, 'iraqi' is close to 'missiles'.