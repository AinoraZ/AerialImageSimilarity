from dataclasses import dataclass
import gc
import numpy as np
import os

from .Models import TrainingStats, SampleTriplet, ImageResult
from ModelBuilders import BaseModelBuilder
from timer import Timer

@dataclass
class TrainingRunnerOptions:
    model_builder: BaseModelBuilder
    sample_count: int

class TrainingRunner:
    def __init__(self, options: TrainingRunnerOptions):
        self.model_builder = options.model_builder
        self.model = self.model_builder.create_model(True)

        self.sample_count = options.sample_count

    def _get_index_choices(self, list_size):
        index_choices_folder = "Experiments/runtime_files/training_samples"
        os.makedirs(index_choices_folder, exist_ok=True)

        index_choices_path = f"{index_choices_folder}/indexes-tc{list_size}-s{self.sample_count}.npy"
        if os.path.isfile(index_choices_path):
            return np.array(sorted(np.load(index_choices_path)))

        index_choices = np.random.choice(list_size, self.sample_count, replace=False)
        np.save(index_choices_path, index_choices)

        return np.array(sorted(index_choices))

    def _fetch_images_from_dataset(self, test_ds, indices):
        image_list = []

        index = 0
        for test_image, _ in test_ds.as_numpy_iterator():
            if index in indices:
                image_list.append(test_image)

            index += 1

        return np.array(image_list)

    def _get_sample_triplets(self, test_ds, results):
        anchors = results[0::3]
        positives = results[1::3]
        negatives = results[2::3]

        triplet_list = np.array(list(zip(anchors, positives, negatives)))
        index_choices = self._get_index_choices(len(triplet_list))

        sample_images = self._fetch_images_from_dataset(test_ds, index_choices)
        sample_triplet_results = triplet_list[index_choices]

        print(sample_images.shape)

        image_results: list[tuple(ImageResult, ImageResult, ImageResult)] = []
        for (sa, sp, sn), (ra, rp, rn) in zip(sample_images, sample_triplet_results):
            image_results.append((
                ImageResult(sa, ra),
                ImageResult(sp, rp),
                ImageResult(sn, rn)
            ))

        triplets: list[SampleTriplet] = []
        for anchor, positive, negative in image_results:
            triplet = SampleTriplet(anchor, positive, negative)
            triplets.append(triplet)

        return triplets

    def _calculate_distances(self, diffs) -> list[float]:
        distances = []
        for diff in diffs:
            max_distance = np.prod(diff.shape)
            distance = np.sqrt(np.nansum(diff) / max_distance)
            
            distances.append(distance)

        return distances

    def run(self, test_dataset) -> TrainingStats:
        print("Calculating training stats...")

        chunk_len = int(len(test_dataset) / 3)
        results = []

        with Timer() as t:
            for s in range(3):
                partial_results = self.model.predict(test_dataset.skip(s * chunk_len).take(chunk_len), verbose = 1,)
                results.extend(partial_results)

        results = np.array(results)
        anchors = results[0::3]
        positives = results[1::3]
        negatives = results[2::3]

        positive_diffs = np.square(anchors - positives)
        positive_distances = self._calculate_distances(positive_diffs)

        negative_diffs = np.square(anchors - negatives)
        negative_distances = self._calculate_distances(negative_diffs)

        #test_triplets = np.array([test[0] for test in list(test_dataset.unbatch().as_numpy_iterator())])
        sample_triplets = self._get_sample_triplets(test_dataset, results)

        training_stats = TrainingStats(
            sample_triplets,
            positive_distances,
            negative_distances,
            t.interval)
        
        gc.collect()

        return training_stats