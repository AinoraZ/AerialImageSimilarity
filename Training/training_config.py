from cgi import test
import tensorflow as tf
import os
import matplotlib.pyplot as plt

from ModelBuilders.model_options import ModelOptions

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from ModelBuilders import BaseModelBuilder

class TrainingConfig:
    def __init__(self, data_folder: str, save_samples: int):
        self.train_dir = os.path.join(data_folder, 'train')
        self.valid_dir = os.path.join(data_folder, 'valid')
        self.test_dir = os.path.join(data_folder, 'test')

        self.save_samples = save_samples

    def __process_images(self, path):    
        image = tf.io.read_file(path)
        image = tf.io.decode_image(image)
        return image, path

    def train_triplet(self, options: ModelOptions):
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = tf.data.Dataset.list_files(str(self.train_dir + '/*/*.png'), shuffle=False)
        train_ds = train_ds.map(self.__process_images, num_parallel_calls=AUTOTUNE).batch(options.batch_size * 3).prefetch(buffer_size=AUTOTUNE)
        return train_ds

    def valid_triplet(self, options: ModelOptions):
        AUTOTUNE = tf.data.AUTOTUNE
        val_ds = tf.data.Dataset.list_files(str(self.valid_dir + '/*/*.png'), shuffle=False)
        val_ds = val_ds.map(self.__process_images, num_parallel_calls=AUTOTUNE).batch(options.batch_size * 3).prefetch(buffer_size=AUTOTUNE)
        return val_ds

    def test_triplet(self):
        AUTOTUNE = tf.data.AUTOTUNE
        test_ds = tf.data.Dataset.list_files(str(self.test_dir + '/*/*.png'), shuffle=False)
        test_ds = test_ds.map(self.__process_images, num_parallel_calls=AUTOTUNE).batch(3).prefetch(buffer_size=AUTOTUNE)

        return test_ds

    def best_weights_path(self, model_builder: BaseModelBuilder):
        base_folder = model_builder.weights_folder()
        return f'{base_folder}/weights.best.hdf5'

    def callbacks(self, model_builder: BaseModelBuilder):
        callbacks = [
            ModelCheckpoint(filepath=self.best_weights_path(model_builder), verbose=1, save_best_only=True, save_weights_only=True, mode='auto'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
            EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True),
        ]

        return callbacks

    def plot_metrics(self, model_builder: BaseModelBuilder, history, metrics=['loss'], skip_start=0.):
        """
        Plots metrics from keras training history.
        """
        hist = history.history
        start_indice = int(len(hist[metrics[0]]) * skip_start)
        
        folder = model_builder.weights_folder()
        representation = model_builder.get_options().representation()

        for metric in metrics:
            plt.plot(hist[metric][start_indice:], label="train {}".format(metric))
            plt.plot(hist[f"val_{metric}"][start_indice:], label=f"val {metric}")
            legend = plt.legend()
            plt.title(metric)
        
            learning_file = f"{folder}/{representation}-learning-{metric}.png"

            plt.savefig(learning_file, bbox_inches='tight', facecolor='w', dpi=300, bbox_extra_artists=(legend,),)
            plt.show()