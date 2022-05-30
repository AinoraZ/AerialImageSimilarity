import statistics
import numpy as np
import json
import os
import matplotlib.pyplot as plt

from PIL import Image
from color_generator import ColorGenerator

from vector import Vector2D
from WeightCalculators.Transformers import BaseTransformer

from .Models import DistanceData, PointData
from WeightCalculators.Transformers import BaseTransformer

class AggregatedData:
    def __init__(self, target_distance: int, distances_dump: list[DistanceData]):
        self.target_distance = target_distance
        self.avg_weight = statistics.mean([distance_data.avg_weight for distance_data in distances_dump])
        self.avg_delta = statistics.mean([distance_data.avg_delta for distance_data in distances_dump])
        self.min_weight = min([distance_data.avg_weight for distance_data in distances_dump])

    def dump(self):
        return {
            "target_distance": self.target_distance,
            "avg_weight": self.avg_weight,
            "avg_delta": self.avg_delta,
            "min_weight": self.min_weight,
        }

class PointGraphGenerator:
    def __init__(self, label: str, y_ticks: list[float], separator_line: float, target_distances: list[int], transformer: BaseTransformer, color_generator: ColorGenerator = None):
        self.label = label
        self.distances = list(np.array([0] + target_distances) * 0.3045634921) #m/px
        self.target_max = max(self.distances)
        self.separator_line = separator_line
        self.y_ticks = y_ticks

        self.transformer = transformer
        self.color_generator = color_generator
        self.colored = color_generator is not None
        self.grouped = False

    def _generate_graph_base(self):
        # plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams['font.size'] = 14

        fig, ax = plt.subplots()
        fig.set_size_inches(8, 5)

        ax.set_xlabel("Atstumas (metrai)")
        ax.set_ylabel("Panašumas")

        ax.set_xlim([0, self.target_max + 1])
        ax.set_xticks(np.arange(0, self.target_max + 1, 25))

        if self.y_ticks is not None:
            ax.set_yticks(self.y_ticks)

        if self.separator_line is not None:
            ax.plot([0, self.target_max], [self.separator_line, self.separator_line], color='r', label="panašumo riba", alpha=0.8)

        return fig, ax

    def generate_distance_graph(self, data: PointData):
        fig, ax = self._generate_graph_base()
        transformer = self.transformer

        weight_data = [transformer.transform(data.original_weight)] + [transformer.transform(distance.avg_weight) for distance in data.distances]
        ax.plot(self.distances, weight_data, marker="o")

        return fig, ax

    def generate_aggregated_graph(self, point_dump: list[PointData], aggregated_dump: list[AggregatedData]):
        fig, ax = self._generate_graph_base()
        transformer = self.transformer

        grouped_point_dump : dict[str, list[PointData]] = {}
        for point_data in point_dump:
            label = point_data.reference_point.label
            if label not in grouped_point_dump:
                grouped_point_dump[label] = []
            
            grouped_point_dump[label].append(point_data)

        if self.grouped:
            for label, point_data_group in grouped_point_dump.items():
                if self.colored:
                    label = label
                    data_color = self.color_generator.label_to_color(label)
                    alpha = 0.40
                else:
                    label = "runs"
                    data_color = "k"
                    alpha = 0.15

                weight_data = \
                    [transformer.transform(point_data_group[0].original_weight)] + \
                    [transformer.transform(distance.avg_weight) for distance in point_data_group[0].distances]

                for point_data in point_data_group[1:]:
                    individual_weight_data = \
                        [transformer.transform(point_data.original_weight)] + \
                        [transformer.transform(distance.avg_weight) for distance in point_data.distances]

                    for index, w in enumerate(individual_weight_data):
                        weight_data[index] += w

                for index, w in enumerate(weight_data):
                    weight_data[index] /= len(point_data_group)

                ax.plot(self.distances, weight_data, color=data_color, alpha=alpha, label=label)
        else:
            for point_data in point_dump:
                if self.colored:
                    label = point_data.reference_point.label
                    data_color = self.color_generator.label_to_color(label)
                    alpha = 0.40
                else:
                    label = "runs"
                    data_color = "k"
                    alpha = 0.15

                weight_data = \
                    [transformer.transform(point_data.original_weight)] + \
                    [transformer.transform(distance.avg_weight) for distance in point_data.distances]

                ax.plot(self.distances, weight_data, color=data_color, alpha=alpha, label=label)

        average_reference_weight = statistics.mean([point_data.original_weight for point_data in point_dump])
        total_weight_data = [transformer.transform(average_reference_weight)] + [transformer.transform(aggregate.avg_weight) for aggregate in aggregated_dump]
        
        ax.plot(self.distances, total_weight_data, marker='o', label="vidurkis")

        handles, labels = ax.get_legend_handles_labels()

        by_label = dict(zip(labels, handles))
        labels = [f"{label}" if label != "vidurkis" else label for label in by_label.keys()]

        lgd = ax.legend(by_label.values(), labels, loc='center left', bbox_to_anchor=(1, 0.5))

        return fig, ax, lgd

    def save(self, fig, filename, legend = None):
        if legend is None:
            return fig.savefig(filename, bbox_inches='tight', facecolor='w', dpi=300)

        return fig.savefig(filename, bbox_inches='tight', facecolor='w', dpi=300, bbox_extra_artists=(legend,),)

class OffsetDataSaver:
    def __init__(self, base_folder: str, target_distances: list[int], color_generator: ColorGenerator = None, separator_line: float = None):
        self.run_folder = f"{base_folder}/Offsets"
        self.target_distances = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100, 150, 250, 350, 500, 700, 1000]

        self.linear_graph_generator = PointGraphGenerator(
            label="linear",
            y_ticks=np.arange(0, 1.01, 0.05),
            separator_line=separator_line, 
            target_distances=target_distances,
            transformer=BaseTransformer(),
            color_generator=color_generator)

    def _build_point_folder(self, point: Vector2D):
        point_dir = f"{self.run_folder}/{point}"
        os.makedirs(point_dir, exist_ok=True)

        return point_dir

    def _save_stats(self, point_folder: str, data: PointData):
        point_file = f"{point_folder}/point-stats.json"

        with open(point_file, "w") as file:
            file.write(json.dumps(data.dump(), indent=4))

    def _save_images(self, point_folder: str, data: PointData):
        images_folder = f"{point_folder}/images"
        os.makedirs(images_folder, exist_ok=True)

        Image.fromarray(data.anchor_image).save(f"{images_folder}/anchor.jpg")

        for distance in data.distances:
            distance_folder = f"{images_folder}/{distance.target_distance}"
            os.makedirs(distance_folder, exist_ok=True)

            for offset in distance.offsets:
                file_name = f"{distance_folder}/{offset.vector_offset}.jpg"
                Image.fromarray(offset.offset_image).save(file_name)

    def _save_individual_graphs(self, graph_generator: PointGraphGenerator, point_folder: str, data: PointData):
        fig, ax = graph_generator.generate_distance_graph(data)

        graph_generator.save(fig, f"{point_folder}/{graph_generator.label}_regular_graph.png")

        ax.set_xlim([0, 150])
        ax.set_xticks(np.arange(0, 151, 5))

        graph_generator.save(fig, f"{point_folder}/{graph_generator.label}_zoomed_graph.png")

        plt.close(fig)

    def _save_individual(self, data: PointData):
        point_folder = self._build_point_folder(data.reference_point)
        
        self._save_stats(point_folder, data)
        # self._save_images(point_folder, data)
        self._save_individual_graphs(self.linear_graph_generator, point_folder, data)

    def save_all_individual(self, point_dump: list[PointData]):
        for point_data in point_dump:
            self._save_individual(point_data)

    def _get_aggregated_dump(self, point_dump: list[PointData]) -> list[AggregatedData]:
        aggregated_distances: dict[int, list[DistanceData]] = {}

        for point_data in point_dump:
            for distance_data in point_data.distances:
                target_distance = distance_data.target_distance
                if target_distance not in aggregated_distances:
                    aggregated_distances[target_distance] = []

                aggregated_distances[target_distance].append(distance_data)

        aggregated_dump =  [AggregatedData(target_distance, dumps) for target_distance, dumps in aggregated_distances.items()]

        return aggregated_dump

    def _save_aggregated(self, point_dump: list[PointData], aggregated_dump: list[AggregatedData]):
        all_run_data = {
            "avg_reference_weight": statistics.mean([point_data.original_weight for point_data in point_dump]),
            "targets": [aggregate.dump() for aggregate in aggregated_dump],
        }

        aggregate_file = f"{self.run_folder}/run-stats.json"

        with open(aggregate_file, "w") as file:
            file.write(json.dumps(all_run_data, indent=4))

    def _save_aggregated_graphs(self, graph_generator: PointGraphGenerator, point_dump: PointData, aggregated_dump: list[AggregatedData]):
        fig, ax, lgd = graph_generator.generate_aggregated_graph(point_dump, aggregated_dump)
        graph_generator.save(fig, f"{self.run_folder}/{graph_generator.label}_regular_graph.png", lgd)

        ax.set_xlim([0, 150])
        ax.set_xticks(np.arange(0, 151, 5))

        graph_generator.save(fig, f"{self.run_folder}/{graph_generator.label}_zoomed_graph.png", lgd)

        plt.close(fig)

    def save_all_aggregated(self, point_dump: list[PointData]):
        aggregated_dump =  self._get_aggregated_dump(point_dump)
        
        self._save_aggregated(point_dump, aggregated_dump)
        self._save_aggregated_graphs(self.linear_graph_generator, point_dump, aggregated_dump)

    def save(self, point_dump: list[PointData]):
        print("Saving offset data")
        self.save_all_individual(point_dump)
        self.save_all_aggregated(point_dump)