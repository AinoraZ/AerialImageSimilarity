import os

from numpy import dtype
import tensorflow as tf
from . import ModelOptions

from tensorflow.keras import backend
from tensorflow.keras import backend, applications, optimizers
from tensorflow.keras.layers import Input, Dropout, Flatten, Dense, Reshape, BatchNormalization, LSTM, Activation
from tensorflow.keras.callbacks import Callback, TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.api._v2.keras import Model

class BaseModelBuilder:
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

    def _pd_alt(self):
        def pd_alt(y_true, y_pred):
            anchor = y_pred[0::3]
            positive = y_pred[1::3]
            positive_distance = tf.reduce_sum(input_tensor=tf.square(tf.subtract(anchor, positive)), axis=0)
            return backend.mean(positive_distance)

        return pd_alt

    def _nd_alt(self):
        def nd_alt(y_true, y_pred):
            anchor = y_pred[0::3]
            negative = y_pred[2::3]
            negative_distance = tf.reduce_sum(input_tensor=tf.square(tf.subtract(anchor, negative)), axis=0)
            return backend.mean(negative_distance)

        return nd_alt

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

    def _generate_model(self, network, load_weights = False):
        lt1 = Dense(self.ingest_dense, activation = 'sigmoid')
        lt2 = Dropout(0.5)
        lt3 = Dense(self.output_dense, dtype="float32")
        lt4 = Activation('sigmoid', dtype='float32', name="prediction")

        lt1_a = lt1(network.layers[self.model_nn].output)
        lt2_a = lt2(lt1_a)
        lt3_a = lt3(lt2_a)
        lt4_a = lt4(lt3_a)

        model: Model = tf.keras.models.Model(inputs = [network.input], outputs = lt4_a)

        for index, layer in enumerate(model.layers):
            if 'dense' in layer.name or index >= self.trainable_from_index:
                break

            layer.trainable = False
        
        if load_weights:
            model.load_weights(self.weights_path())

        model.compile(optimizer = optimizers.Adam(), loss = self._triplet_loss(), metrics = [self._pd(), self._nd(), self._pd_alt(), self._nd_alt()])

        return model

    def get_options(self) -> ModelOptions:
        pass

    def create_model(self, load_weights = False) -> Model:
        pass