import tensorflow as tf
import keras

class EvenLayer(keras.layers.Layer):
    def call(self, inputs):
        return tf.math.sin(inputs)