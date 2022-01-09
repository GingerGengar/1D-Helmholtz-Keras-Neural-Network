import numpy as np
import keras
import tensorflow as tf

#Custom Define a layer acting as even function
class EvenLayer(keras.layers.Layer):
    def call(self, inputs):
        return tf.math.cos(inputs)

#Custom Define a layer acting as odd function
class OddLayer(keras.layers.Layer):
    def call(self, inputs):
        return tf.math.sin(inputs)

#Custom Define a layer acting as multiplication operation
class MultLayer(keras.layers.Layer):
    def call(self, input1, input2):
        return tf.math.multiply(input1, input2)

#Defining Inputs of Neural Network
inputs = tf.keras.Input(shape=(1,))

#Implementing the even function $f(x)$,
layer0a = EvenLayer()(inputs)

#Some Arbitrary Neural Network Architecture
layer1a = tf.keras.layers.Dense(4, activation='sigmoid')(layer0a)
layer2a = tf.keras.layers.Dense(4, activation='sigmoid')(layer1a)
layer3a = tf.keras.layers.Dense(1, activation='sigmoid')(layer1a)

#Implementing the Odd function $g(x)$
layer0b = OddLayer()(inputs)

#Defining Outputs of Neural Network
outputs = MultLayer()(layer3a,layer0b)

#Defining Model used
model = tf.keras.Model(inputs = inputs, outputs = outputs)

#Testing the model
test = np.array([-4,-3,-2,-1, 0, 1, 2, 3, 4])
print(model(test))

#Customary End
print('Leaves Blow in the Wind...')
