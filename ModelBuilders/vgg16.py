from numpy import dtype
import tensorflow as tf
import os

from tensorflow.keras import backend, applications, optimizers
from tensorflow.keras.layers import Input, Dropout, Flatten, Dense, Reshape, BatchNormalization, LSTM, Activation
from tensorflow.keras.callbacks import Callback, TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.api._v2.keras import Model

from . import BaseModelBuilder, ModelOptions

default_options = ModelOptions(
    builder_label="VGG16",
    model_nn=18,
    ingest_dense=64,
    output_dense=8,
    trainable_from_index=18,
    epochs=6,
    batch_size=1,
)

class Vgg16NetBuilder(BaseModelBuilder):
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
        mobilenet = applications.VGG16(include_top = False, weights = 'imagenet', input_shape=(224, 224, 3))
        
        return self._generate_model(mobilenet, load_weights)