import os
import math
from PIL import Image
import json

from ExperimentRunners.TrainingRunner.Models import ImageResult
from .Models import TrainingStats
import matplotlib.pyplot as plt

class TrainingSaver:
    def __init__(self, base_folder):
        self.run_folder = f"{base_folder}/training"
        os.makedirs(self.run_folder, exist_ok=True)

    def __sample_triplet_folder(self, index):
        sample_folder = f"{self.run_folder}/{index}"
        os.makedirs(sample_folder, exist_ok=True)

        return sample_folder

    def __save_image_result(self, sample_folder, label, image_result: ImageResult):
        image = Image.fromarray(image_result.image)
        image_name = f"{sample_folder}/image-{label}.png"

        image.save(image_name)

        embedding_name = f"{sample_folder}/emb-{label}.png"

        if image_result.embedding.shape[-1] == 1:
            plt.imshow(image_result.embedding[:,:,0])
            plt.savefig(embedding_name, bbox_inches='tight', facecolor='w', dpi=300)
        else:
            rows = 2
            columns = math.ceil(image_result.embedding.shape[-1] / rows)

            fig, axs = plt.subplots(rows, columns, sharex=True, sharey=True)

            for index, ax in enumerate(axs.flat):
                ax.imshow(image_result.embedding[:,:,index])

            fig.savefig(embedding_name, bbox_inches='tight', facecolor='w', dpi=300)

            plt.close(fig)

    def _save_sample_triplets(self, training_stats: TrainingStats):
        print("Saving sampled data", end="")
        for index, sample in enumerate(training_stats.sample_triplets):
            print(".", end="")
            sample_folder = self.__sample_triplet_folder(index)

            self.__save_image_result(sample_folder, "anchor", sample.anchor)
            self.__save_image_result(sample_folder, "positive", sample.positive)
            self.__save_image_result(sample_folder, "negative", sample.negative)

        print()

    def _save_distribution_graph(self, training_stats: TrainingStats):
        plt.scatter(range(0, len(training_stats.positive_distances)), training_stats.positive_distances, alpha=0.3, label="ta")
        plt.scatter(range(0, len(training_stats.negative_distances)), training_stats.negative_distances, alpha=0.3, label="na")
        legend = plt.legend()
        
        plt.title("Panašumas")

        filename = f"{self.run_folder}/distribution.png"
        plt.savefig(filename, bbox_inches='tight', facecolor='w', dpi=300, bbox_extra_artists=(legend,),)

        plt.show()

    def _save_stats(self, training_stats: TrainingStats):
        point_file = f"{self.run_folder}/recommendation-stats.json"

        with open(point_file, "w") as file:
            file.write(json.dumps(training_stats.dump(), indent=4))

    def save(self, training_stats: TrainingStats):
        self._save_stats(training_stats)
        self._save_distribution_graph(training_stats)
        self._save_sample_triplets(training_stats)
