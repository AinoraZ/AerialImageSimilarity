import sys
import tensorflow as tf
import os

from keras.api._v2.keras import Model

from tensorflow.keras import backend, optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, Flatten, Dense, Reshape, BatchNormalization, LSTM, Activation
import efficientnet.tfkeras as efn 

from . import BaseModelBuilder, ModelOptions

default_options = ModelOptions(
    builder_label="CustomEfficientNetB2",
    model_nn=59,
    ingest_dense=64,
    output_dense=8,
    trainable_from_index=sys.maxsize,
    epochs=4,
    batch_size=1,
)

class CustomEfficientNetB2Builder(BaseModelBuilder):
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

    def _pretrained_model(self):
        input_tensor = Input(shape=(224, 224, 3))
        base_model = efn.EfficientNetB2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        
        bn = BatchNormalization()(input_tensor)
        x = base_model(bn)
        x = Flatten()(x)
        output = Dense(17, activation='sigmoid')(x)
        model = Model(input_tensor, output)
        
        model.load_weights(f"{self.base_folder}/amazon_pretrain/b2.weights.best.hdf5")
        
        return model

    def get_options(self) -> ModelOptions:
        return self.options

    def create_model(self, load_weights = False):
        pretrained_model = self._pretrained_model()

        base_model = efn.EfficientNetB2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.set_weights(pretrained_model.layers[2].get_weights())

        lt1 = Dense(self.ingest_dense, activation='sigmoid')
        lt2 = Dropout(0.5)
        lt3 = Dense(self.output_dense, activation = 'sigmoid')
        lt4 = Activation('sigmoid', dtype='float32', name="prediction")

        lt1_a = lt1(base_model.layers[self.model_nn].output)
        lt2_a = lt2(lt1_a)
        lt3_a = lt3(lt2_a)
        lt4_a = lt4(lt3_a)

        model = Model(base_model.input, lt4_a)

        for index, layer in enumerate(model.layers):
            if 'dense' in layer.name or index >= self.trainable_from_index:
                break

            layer.trainable = False

        if load_weights:
            model.load_weights(self.weights_path())

        model.compile(optimizer = optimizers.Adam(), loss = self._triplet_loss(), metrics = [self._pd(), self._nd(), self._pd_alt(), self._nd_alt()])

        return model