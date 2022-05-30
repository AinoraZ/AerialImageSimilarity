from dataclasses import dataclass
import os
import numpy as np

from map_provider import ImageProjection, MapProvider

@dataclass
class RecallConfig:
    location_count: int
    projection: ImageProjection
    crop_size: int
    save_samples: tuple[int, int]
    top: int
    step_size: int
    relevant_distance: int

    def load_recall_locations(self, drone_provider: MapProvider):
        location_count = self.location_count
        save_folder = "Experiments/runtime_files/recall-locations"
        os.makedirs(save_folder, exist_ok=True)

        file = f"{save_folder}/size{location_count}-{drone_provider.projection}.npy"
        if os.path.isfile(file):
            return np.load(file)

        locations = drone_provider.generate_random_locations(location_count)
        np.save(file, locations)
        
        return locations