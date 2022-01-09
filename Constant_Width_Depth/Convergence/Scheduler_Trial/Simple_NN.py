import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

#Training Numbers
trainum = 1000

#Training Inputs
traininputs = np.array([
    [1.0,0.0,0.0],
    [0.0,1.0,0.0],
    [0.0,0.0,1.0]
])

#Training Outputs
trainoutputs = np.array([
    [0.0,1.0,0.0],
    [1.0,0.0,0.0],
    [0.0,0.0,1.0]
])

#Defining Inputs of Neural Network
inputs = tf.keras.Input(shape=(3))

#Some Arbitrary Neural Network Architecture
layer1 = tf.keras.layers.Dense(4, activation='sigmoid')(inputs)
layer2 = tf.keras.layers.Dense(4, activation='sigmoid')(layer1)

#Defining Outputs of Neural Network
outputs = tf.keras.layers.Dense(3)(layer2)

#Defining Model used
model = tf.keras.Model(inputs = inputs, outputs = outputs)

#Determine the Learning Rate
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.2)

#Show the Decaying Learning Rate
def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer._decayed_lr(tf.float32)
    return lr

#Instantiate an optimizer
opt = tf.keras.optimizers.Adam(learning_rate = lr_schedule)
lr_metric = get_lr_metric(opt)

#Compile Model
model.compile(
    optimizer = opt,
    metrics=['accuracy', lr_metric],
    loss='mean_squared_error')

#Train Model
model.fit(
    x=traininputs, 
    y=trainoutputs, 
    batch_size=None,
    epochs = trainum
)

#Testing the model
print(model(traininputs))

#Customary End
print('Leaves Blow in the Wind...')
