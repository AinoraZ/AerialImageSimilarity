from __future__ import annotations
import numpy as np
from datetime import datetime
import gc
from tensorflow.python.keras.callbacks import TensorBoard

from ModelBuilders import BaseModelBuilder
from WeightCalculators.Transformers import BaseTransformer

from WeightCalculators import WeightCalculator

class ModelBasedWeightCalculator(WeightCalculator):
    def __init__(self, model_builder: BaseModelBuilder, batch_size, transformer: BaseTransformer):
        self.model_builder = model_builder
        self.model = model_builder.create_model(True)
        self.batch_size = batch_size

        logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.board_callback = TensorBoard(
            log_dir = logs,
            histogram_freq = 1,
            update_freq=1,
            profile_batch = 32)

        self.transformer = transformer

    def create_weights_from_images(self, anchor_image, images) -> list:
        #with tf.profiler.experimental.Profile("logs/"):
            #pass

        drone_emb = self.model.predict(np.array([anchor_image]), batch_size=1)
        reference_embs = self.model.predict(np.array(images), batch_size=self.batch_size)

        weights = self.calculate_weights(drone_emb, reference_embs)

        gc.collect()

        return [self.transformer.transform(weight) for weight in weights]

    def calculate_weights(self, anchor_emb, reference_embs):
        max_distance = np.prod(anchor_emb.shape)
        all_distances = np.square(reference_embs - anchor_emb)

        weights = []

        for pair_distance in all_distances:
            distance = np.nansum(pair_distance)

            normalized_distance = 1 - np.sqrt((distance / max_distance)) # 1-similar, 0-different
            weights.append(normalized_distance)

        return weights