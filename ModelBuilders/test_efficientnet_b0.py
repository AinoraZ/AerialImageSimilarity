from numpy import dtype
import tensorflow as tf
import os
import sys

from tensorflow.keras import backend, applications, optimizers
from tensorflow.keras.layers import Input, Dropout, Flatten, Dense, Reshape, BatchNormalization, LSTM, Activation, Lambda
from tensorflow.keras.callbacks import Callback, TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.api._v2.keras import Model

from . import TestBaseModelBuilder, ModelOptions

default_options = ModelOptions(
    builder_label="Experimental/NewTraining/EfficientNetV2B0",
    model_nn=30,
    ingest_dense=64,
    output_dense=8,
    trainable_from_index=30,
    epochs=6,
    batch_size=16,
)

class TestEfficientNetB0Builder(TestBaseModelBuilder):
    def __init__(self, options: ModelOptions = None):
        if options is None:
            options = default_options

        self.options = options

        self.builder_label = options.builder_label
        self.base_folder = self.weights_folder()

        self.model_nn = options.model_nn
        self.ingest_dense = options.ingest_dense
        self.output_dense = options.output_dense
        self.trainable_from_index = options.trainable_from_index

        self.epsilon = 1e-6

    def get_options(self) -> ModelOptions:
        return self.options

    def _training_model(self) -> Model:
        mobilenet = applications.EfficientNetV2B0(include_top = False, weights = 'imagenet', input_shape=(224, 224, 3))
        
        lt1 = Dense(self.ingest_dense, activation = 'sigmoid')
        lt2 = Dropout(0.5)
        lt3 = Dense(self.output_dense, dtype="float32", activation=None)
        lt4 = Flatten()
        lt5 = Lambda(lambda x: tf.math.l2_normalize(x, axis=1))

        lt1_a = lt1(mobilenet.layers[self.model_nn].output)
        lt2_a = lt2(lt1_a)
        lt3_a = lt3(lt2_a)
        lt4_a = lt4(lt3_a)
        lt5_a = lt5(lt4_a)

        model: Model = tf.keras.models.Model(inputs = [mobilenet.input], outputs = lt5_a)

        for index, layer in enumerate(model.layers):
            if 'dense' in layer.name or index >= self.trainable_from_index:
                break

            layer.trainable = False
        
        model.compile(optimizer = optimizers.Adam(), loss = self._triplet_loss(), metrics = [self._pd(), self._nd()])

        return model

    def _prediction_model(self) -> Model:
        mobilenet = applications.EfficientNetV2B0(include_top = False, weights = 'imagenet', input_shape=(224, 224, 3))
        
        lt1 = Dense(self.ingest_dense, activation = 'sigmoid')
        lt2 = Dropout(0.5)
        lt3 = Dense(self.output_dense, dtype="float32")
        lt4 = Activation('sigmoid', dtype='float32', name="prediction")

        lt1_a = lt1(mobilenet.layers[self.model_nn].output)
        lt2_a = lt2(lt1_a)
        lt3_a = lt3(lt2_a)
        lt4_a = lt4(lt3_a)

        model: Model = tf.keras.models.Model(inputs = [mobilenet.input], outputs = lt4_a)

        for index, layer in enumerate(model.layers):
            if 'dense' in layer.name:
                break

            layer.trainable = False

        model.load_weights(self.weights_path())
        
        model.compile(optimizer = optimizers.Adam(), loss = self._triplet_loss(), metrics = [self._pd(), self._nd()])

        return model

    def create_model(self, load_weights = False) -> Model:
        if load_weights:
            return self._prediction_model()
        
        return self._training_model()