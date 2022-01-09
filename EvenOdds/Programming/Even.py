import numpy as np
import keras
import tensorflow as tf

#Custom Define a layer acting as even function
class EvenLayer(keras.layers.Layer):
    def call(self, inputs):
        return tf.math.cos(inputs)

#Defining Inputs of Neural Network
inputs = tf.keras.Input(shape=(1,))

#Implementing the efen function $f(x)$,
layer0 = EvenLayer()(inputs)

#Some Arbitrary Neural Network Architecture
layer1 = tf.keras.layers.Dense(4, activation='sigmoid')(layer0)
layer2 = tf.keras.layers.Dense(4, activation='sigmoid')(layer1)

#Defining Outputs of Neural Network
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(layer2)

#Defining Model Used
model = tf.keras.Model(inputs = inputs, outputs = outputs)

#Testing the model
test = np.array([-4,-3,-2,-1, 0, 1, 2, 3, 4])
print(model(test))

#Customary End
print('Leaves Blow in the Wind...')
