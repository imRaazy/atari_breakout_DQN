import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model

class DQNModel():
  @staticmethod
  def create(input_shape, outputs):
    return keras.Sequential([
      keras.layers.Conv2D(32, 8, strides=4, activation="relu", input_shape=input_shape),
      keras.layers.Conv2D(64, 4, strides=2, activation="relu"),
      keras.layers.Conv2D(64, 3, strides=1, activation="relu"),
      keras.layers.Flatten(),
      keras.layers.Dense(512, activation = "relu"),
      keras.layers.Dense(outputs)])
      
  @staticmethod
  def create_duelling(input_shape, output):
    input_layer = keras.layers.Input(shape=input_shape)
    conv = keras.layers.Conv2D(32, 8, strides=4, activation='relu')(input_layer)
    conv = keras.layers.Conv2D(64, 4, strides=2, activation='relu')(conv)
    conv = keras.layers.Conv2D(64, 3, strides=1, activation='relu')(conv)
    flatten = keras.layers.Flatten()(conv)

    pre_v = keras.layers.Dense(512, activation='relu')(flatten)
    v = keras.layers.Dense(1)(pre_v)

    pre_a = keras.layers.Dense(512, activation='relu')(flatten)
    a = keras.layers.Dense(output)(pre_a)

    q = keras.layers.Lambda(lambda args: args[0] + (args[1] - tf.math.reduce_mean(args[1], axis=1, keepdims=True)))([v, a])
    return keras.Model(inputs=[input_layer], outputs=[q])
  
  @staticmethod
  def print_model(model):
    display(plot_model(model, show_shapes=True, show_layer_names=False, rankdir='LR', dpi=92))