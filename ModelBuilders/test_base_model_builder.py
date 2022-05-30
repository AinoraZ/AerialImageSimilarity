from keras.api._v2.keras import Model
import os

from numpy import dtype
import tensorflow as tf
from . import ModelOptions
from tensorflow.keras import backend

class TestBaseModelBuilder:
    def _triplet_loss(self):
        def triplet_loss(y_true, y_pred):
            alpha = 0.5

            anchor = y_pred[0::3]
            positive = y_pred[1::3]
            negative = y_pred[2::3]

            positive_distance = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(anchor, positive), 2), 1, keepdims=True))
            negative_distance = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(anchor, negative), 2), 1, keepdims=True))
            
            loss = tf.reduce_mean(tf.maximum(positive_distance - negative_distance + alpha, 0))
            return loss

        return triplet_loss

    def _pd(self):
        def pd(y_true, y_pred):
            anchor = y_pred[0::3]
            positive = y_pred[1::3]
            positive_distance = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(anchor, positive), 2), 1, keepdims=True))
            
            return backend.mean(positive_distance)

        return pd

    def _nd(self):
        def nd(y_true, y_pred):
            anchor = y_pred[0::3]
            negative = y_pred[2::3]
            negative_distance = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(anchor, negative), 2), 1, keepdims=True))
            return backend.mean(negative_distance)

        return nd

    def weights_file(self) -> str:
        representation = self.get_options().representation()

        return f'{representation}.h5'

    def weights_folder(self) -> str:
        label = self.get_options().builder_label

        folder = f"ModelBuilders/runtime_files/{label}"
        os.makedirs(folder, exist_ok=True)

        return folder

    def weights_path(self) -> str:
        folder = self.weights_folder()
        name = self.weights_file()

        return f"{folder}/{name}"

    def get_options(self) -> ModelOptions:
        pass

    def create_model(self, load_weights = False) -> Model:
        pass