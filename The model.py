#Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split

#The data
df = pd.read_csv('mushrooms.csv')
x = df.drop(columns=['class'])
y = df['class']

#Get the dummy variables in features
x = pd.get_dummies(x)
#Replace e and p to 0s and 1s
y.replace('p', 0, inplace=True)
y.replace('e', 1, inplace=True)

#converting the datatype
x = x.values.astype('float32')
y = y.values.astype('float32')

#The train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
#The validation train test split
x_train, x_validation,y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2)

#The model in Keras
model = keras.Sequential([keras.layers.Dense(32, input_shape=(117,)),
                          keras.layers.Dense(20, activation=tf.nn.relu),
                          keras.layers.Dense(2, activation='softmax')])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

#fitting
history = model.fit(x_train, y_train, epochs=20, validation_data=(x_validation, y_validation))

#The prediction
x_pred = model.predict(x_test)
per = model.evaluate(x_test, y_test)

#Visualisation
history_dict = history.history

# Checking Overfit
acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()