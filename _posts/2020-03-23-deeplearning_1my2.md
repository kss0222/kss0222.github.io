---
layout: post
title:  "딥러닝 시작하기"
date:   2020-03-23
author: Esther. K
categories: Deep Learning_1
tags: 딥러닝
---

파이썬(Python) 및 텐서플로우(tensoflow) 설치하기


> Convolustional lyaer로 MNIST를 분류하는 모델을 만들고 학습시켜 봅니다.
사용환경은 Google Colab에서 Tensorflow 2.0 version Keras를 이용하였습니다. 
 
이번 시간에는 저번 시간에 햇던 DNN에 이어서 Convolution network를 이용해서 MNIST데이터를 분류해 보도록 하겠습니다. 
전체적으로 비슷하지만 어떤점이 다른지 비교하면서 보시면 좋을듯 합니다. 


##  1. 환경 설정
---

* 먼저 실습에 필요한 tensorflow 2.0을 임포트 해줍니다. 

```python
!pip install tensorflow-gpu==2.0.0-rc1
import tensorflow as tf
import numpy as np
```

## 2. Data Preprocessing
---

### MNIST 데이터 불러오기 

```python
mnist = tf.keras.datasets.mnist
(image_train, label_train), (image_test, label_test) = mnist.load_data()
```

* 불러온 데이터셋의 shape를 확인해 봅니다. 

```python
image_train.shape, label_train.shape, image_test.shape, label_test.shape
```
((60000, 28, 28), (60000,), (10000, 28, 28), (10000,))

### MNIST 이미지 그리기

```python
import matplotlib.pyplot as plt

size = 5

plt.figure(figsize=(1.5*size,1.5*size))
for i in range(size*size):
  plt.subplot(size,size,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.xlabel(label_train[i])
  plt.imshow(image_train[i])
plt.show()
```
![02]({{ site.url }}/assets/dnn-MNIST.PNG)

### 데이터 Reshape 및 정규화 

* 여기까진 지난 시간에 했던 DNN노트와 다른 점이 없습니다. 하지만 Reshape 할때 shape가 달라집니다. 
Dense layer에 Input으로 들어갈 땐 데이터의 shape이 1차원 벡터지만 Convolution layer에서는 3차원 (이미지가로, 이미지세로, 채널 수 )
가 됩니다. MNIST 데이터는 28, 28 크기의 이미지이고 채널 수는 1 이기 때문에 (28,28,1)로 데이터를 reshape 해주겠습니다. 

```python
# 채널 차원 추가
x_train = image_train.reshape(image_train.shape[0],image_train.shape[1],image_train.shape[2],1).astype('float32')
x_test  = image_test.reshape(image_test.shape[0], image_test.shape[1], image_test.shape[2], 1).astype('float32')

y_train = label_train.reshape(label_train.shape[0], 1)
y_test = label_test.reshape(label_test.shape[0], 1)

# 이미지셋 정규화
x_train, x_test = x_train / 255.0, x_test / 255.0
```

* train data에서 validation data를 따로 분리해 줍시다. 

```python
import sklearn
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=1)
```

* 데이터의 차원을 다시 한번 확인해 봅시다. 

```python
x_test_shape = x_test.shape
x_train_shape = x_train.shape
x_val_shape = x_val.shape

print('train:',x_train_shape)
print('val:',x_val_shape)
print('test:',x_test_shape)
```
train: (54000, 28, 28, 1)
val: (6000, 28, 28, 1)
test: (10000, 28, 28, 1)


## 3. 모델 Development
---

* 한 층의 Convolutional layer 후에 Flatten을 하는 아주 간단한 모델을 만들었습니다. 기호(?) 에 맞게 모델을 커스텀하여도 좋습니다. 하지만 MNSIT 데이터는 이 정도 모델 만으로도 우수한 성능을 도출 할 수 있습니다. 

```python
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Input, Activation, BatchNormalization

def make_cnn_model():

    model = tf.keras.Sequential()

    model.add(Input(shape=(28,28,1)))

    model.add(Conv2D(64, (5, 5), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))  
    
    model.add(Flatten())
    
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))  

    model.add(Dense(10))
    model.add(Activation('softmax'))  

    return model

cnn = make_cnn_model()
cnn.summary()
```
![02]({{ site.url }}/assets/cnn-summary.PNG)

## 3. 모델 학습
---

* Model.compile() 함수를 이용해 loss, optimizer, metircs를 정의합니다.

```python

cnn.compile(
  loss='sparse_categorical_crossentropy',
  optimizer='adam',
  metrics=['acc']
)

cnn.fit(
    x_train, y_train, 
    validation_data = (x_val, y_val),
    epochs=5,
)
```

![02]({{ site.url }}/assets/cnn-training.PNG)

* 5번의 epoch로도 98프로가 넘는 accuracy가 나옵니다. 지난시간에 Dense layer 로만 만들었던 모델과 비교하면 훨씬 좋은 성능을 보이는 걸 알 수 있습니다. 

## 4. 모델 확인
---

* Test set을 예측해 보고 그려봅시다.

```python
y_pred = cnn.predict(x_test)
y_pred.shape
```
(10000, 10)

```python
size = 7

plt.figure(figsize=(1.5*size,1.5*size))
for i in range(size*size):
