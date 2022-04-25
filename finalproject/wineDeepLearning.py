
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from matplotlib import rcParams


df = pd.read_csv('winequalityN.csv')
#print(df.sample(5)) ## print sample of 5 rows

df = df.dropna() ## drop samples with missing values

## non-numerical feature 'type' --> binary feature 'isWhite'
df['is_white'] = [1 if type == 'white' else 0 for type in df['type']]

## after being a fly on the wall for lenny / calix conversation
## i added another feature to balance this
df['is_red'] = [1 if type == 'red' else 0 for type in df['type']]

df.drop('type', axis=1, inplace=True)

## one-hot encoding
## feature 'quality' will be used to classify the wines
## rather than 'quality' values being 3 to 9, make it binary.
##  quality >= 6, then good. else, bad
df['is_good_wine'] = [1 if quality >= 6 else 0 for quality in df['quality']]
df.drop('quality', axis=1, inplace=True)
#print(df.head())

## there are almost twice as many good wines as bad
X = df.drop('is_good_wine', axis=1)
y = df['is_good_wine']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=14)

## data scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

## export data as new csv's for feature ranking
#dfy = pd.DataFrame(y_train)
#dfx = pd.DataFrame(X_train_scaled)
#dfx.to_csv('frx.csv')
#dfy.to_csv('fry.csv')
#xdata = pd.DataFrame(X)
#xdata.to_csv('xdata.csv')

## training
## the datset is imbalanced
## so we will evaluate model with multiple metrics

## originally, there were 4 layers
## with a max of 12 nodes
## metrics have increased after adding more layers/nodes
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=50, activation='relu'))
model.add(tf.keras.layers.Dense(units=55, activation='relu'))
model.add(tf.keras.layers.Dense(units=40, activation='relu'))
model.add(tf.keras.layers.Dense(units=30, activation='relu'))
model.add(tf.keras.layers.Dense(units=20, activation='relu'))
model.add(tf.keras.layers.Dense(units=15, activation='relu'))
model.add(tf.keras.layers.Dense(units=10, activation='relu'))
model.add(tf.keras.layers.Dense(units=5,  activation='relu'))
model.add(tf.keras.layers.Dense(units=2,  activation='relu'))
model.add(tf.keras.layers.Dense(units=1,  activation='sigmoid'))

#model.summary()

model.compile(
    optimizer=tf.keras.optimizers.SGD(),
    loss = tf.keras.losses.BinaryCrossentropy(),
    metrics = [tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()]
)

history = model.fit(
    X_train_scaled,
    y_train, 
    epochs = 200,
    validation_data = (X_test_scaled, y_test),
    verbose = 2
)






