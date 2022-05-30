import tensorflow as tf
import os
import sys
from keras.api._v2.keras import Model

from tensorflow.keras import backend, applications, optimizers
from tensorflow.keras.layers import Input, Dropout, Flatten, Dense, Reshape, BatchNormalization, LSTM, Activation

from . import BaseModelBuilder, ModelOptions

default_options = ModelOptions(
    builder_label="EfficientNetB2",
    model_nn=59,
    ingest_dense=64,
    output_dense=8,
    trainable_from_index=sys.maxsize,
    epochs=4,
    batch_size=1,
)

class EfficientNetB0Builder(BaseModelBuilder):
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

    def create_model(self, load_weights = False) -> Model:
        base_model = applications.EfficientNetV2B0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        return self._generate_model(base_model, load_weights)