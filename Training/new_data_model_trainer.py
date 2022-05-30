import gc
import os

from image_provider import ImageProvider
from map_provider import ImageProjection, MapProvider

from ExperimentRunners.RecallRunner import RecallRunnerOptions, RecalRunner, RecallSaverOptions, RecallSaver
from ExperimentRunners.TrainingRunner import TrainingRunnerOptions, TrainingRunner, TrainingSaver
from ExperimentRunners.OffsetRunner import OffsetRunnerOptions, OffsetRunner, OffsetDataSaver

from vector import Vector2D
from ModelBuilders import BaseModelBuilder
from . import TrainingConfig, OffsetConfig, RecallConfig

class NewDataModelTrainer:
    def __init__(self):
        self.recall_city_image_path = "City/NewTraining/ExperimentZone/City_2017.jpg"
        self.recall_drone_image_path = "City/NewTraining/ExperimentZone/City_2016.jpg"
        self.offset_city_image_path = "City/NewCut/City_2017.jpg"
        self.offset_drone_image_path = "City/NewCut/City_2016.jpg"

        self.training_config = TrainingConfig(
            data_folder="City/NewTraining",
            save_samples=10,
        )

        self.offset_config = OffsetConfig(
            drone_crop_size=672,
            city_crop_size=672,
        )

        self.recall_config = RecallConfig(
            location_count=1000,
            projection=ImageProjection(position=Vector2D(524, 524), size=Vector2D(5000, 5000)),
            crop_size=672,
            save_samples=(4, 4),
            top=5,
            step_size=25,
            relevant_distance=50,
        )

    def _train_model(self, model_builder: BaseModelBuilder):
        weights_file = model_builder.weights_path()
        if os.path.isfile(weights_file):
            print("Skipped training, weight file already exists...")
            return

        options = model_builder.get_options()
        model = model_builder.create_model()

        history = model.fit(
            self.training_config.train_triplet(options), 
            epochs = options.epochs, 
            validation_data = self.training_config.valid_triplet(options), 
            callbacks = self.training_config.callbacks(model_builder),
            shuffle = False)

        self.training_config.plot_metrics(model_builder, history)

        model.load_weights(self.training_config.best_weights_path(model_builder))
        model.save(model_builder.weights_path())

        model = None
        gc.collect()

    def _experiment_folder(self, model_builder: BaseModelBuilder):
        options = model_builder.get_options()
        label = options.builder_label
        representation = options.representation()

        experiment_folder = f"Data/{label}/{representation}"
        os.makedirs(experiment_folder, exist_ok=True)

        return experiment_folder

    def _training_experiment(self, model_builder: BaseModelBuilder):
        options = TrainingRunnerOptions(model_builder, self.training_config.save_samples)
        training_runner = TrainingRunner(options)

        stats = training_runner.run(self.training_config.test_triplet())
        training_saver = TrainingSaver(self._experiment_folder(model_builder))

        training_saver.save(stats)

    def _offset_experiment(self, model_builder: BaseModelBuilder):
        city_image = ImageProvider(image_path=self.offset_city_image_path)
        drone_image = ImageProvider(image_path=self.offset_drone_image_path)

        city_provider = MapProvider(
            image_provider=city_image,
            crop_size=self.offset_config.city_crop_size,
            projection=None)

        drone_provider = MapProvider(
            image_provider=drone_image,
            crop_size=self.offset_config.drone_crop_size,
            projection=None)

        city_image.close()
        drone_image.close()
        gc.collect()

        options = OffsetRunnerOptions(
            model_builder=model_builder,
            city_map=city_provider,
            drone_map=drone_provider,
            reference_points=self.offset_config.generate_reference_points())

        offset_runner = OffsetRunner(options)
        stats = offset_runner.run()

        offset_saver = OffsetDataSaver(
            base_folder=self._experiment_folder(model_builder),
            target_distances=offset_runner.target_distances,
            color_generator=self.offset_config.build_color_generator())

        offset_saver.save(stats)

    def _recall_experiment(self, model_builder: BaseModelBuilder):
        city_image = ImageProvider(image_path=self.recall_city_image_path)
        drone_image = ImageProvider(image_path=self.recall_drone_image_path)

        city_provider = MapProvider(
            image_provider=city_image,
            crop_size=self.recall_config.crop_size,
            projection=self.recall_config.projection)

        drone_provider = MapProvider(
            image_provider=drone_image,
            crop_size=self.recall_config.crop_size,
            projection=self.recall_config.projection)

        city_image.close()
        drone_image.close()
        gc.collect()

        runner_options = RecallRunnerOptions(
            model_builder=model_builder,
            city_map=city_provider,
            drone_map=drone_provider,
            top=self.recall_config.top,
            step_size=self.recall_config.step_size,
            relevant_distance=self.recall_config.relevant_distance,
            drone_locations=self.recall_config.load_recall_locations(drone_provider))

        runner = RecalRunner(runner_options)
        stats = runner.run()

        saver_options = RecallSaverOptions(
            city_map=city_provider,
            drone_map=drone_provider,
            save_samples=self.recall_config.save_samples)

        saver = RecallSaver(self._experiment_folder(model_builder), saver_options)
        saver.save(stats)

    def _ran_to_completion_file(self, model_builder: BaseModelBuilder) -> str:
        experiment = self._experiment_folder(model_builder)
        return f"{experiment}/.complete"

    def _ran_to_completion(self, model_builder: BaseModelBuilder) -> bool:
        return os.path.isfile(self._ran_to_completion_file(model_builder))

    def _mark_complete(self, model_builder: BaseModelBuilder):
        open(self._ran_to_completion_file(model_builder), mode="a").close()

    def _run_experiments(self, model_builder: BaseModelBuilder):
        self._training_experiment(model_builder)
        self._offset_experiment(model_builder)   
        self._recall_experiment(model_builder)

        self._mark_complete(model_builder)

    def run(self, model_builder: BaseModelBuilder, run_experiments = True):
        options = model_builder.get_options()
        print("Now testing...", options.builder_label, options.representation())
        
        if self._ran_to_completion(model_builder):
            print("Experiments already complete. Remove the '.complete' file for re-run")
            return

        self._train_model(model_builder)

        if run_experiments:
            self._run_experiments(model_builder)
