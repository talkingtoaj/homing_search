from sklearn.model_selection import train_test_split
from pandas import DataFrame
import tensorflow as tf
from tensorflow.keras.optimizers import schedules
from .utils import add_to_log, blank_log, moving_average, sort_dict_keys_alphabetically, remove_nan_results, unique_pairs
from .data import KerasAdaptor, SklearnAdaptor

class AbstractAdaptor():
    def __init__(self, data, label, val_pc = 0.2):
        data_type = type(data)
        if data_type is DataFrame:
            self.load_pandas(data)
        else:
            raise NotImplemented(f"data input of type {data_type} is not supported")
        self.label = label
        self.val_pc = 0.2

    def load_pandas(self, data:DataFrame):
        # TODO: Modify to accept a list of labels
        self.df = data

    def prepare_data(self, batch_size):
        """ splits, shuffles and converts data from source dataset """
        self.get_train_test_sets()
        self.convert_pandas(batch_size)

    def get_train_test_sets(self):
        self.train_df, self.val_df = train_test_split(self.df, test_size=self.val_pc, shuffle=True)

    def convert_pandas(self, batch_size):
        """ convert pandas Dataframe into data structure suited to target API """
        raise NotImplemented

    def fit(self, batch_size=256):
        # super().prepare_data(batch_size)
        # run native fit
        # return metrics
        raise NotImplemented

    def sub_epochs(self, fraction) -> int:
        return round(self.epochs*fraction)

class KerasAdaptor(AbstractAdaptor):
    def __init__(self, save_tf_logs=False):
        super().__init__()        
        self.save_tf_logs = save_tf_logs

    def convert_pandas(self, batch_size):
        self.train_ds = self.df_to_dataset(self.train_df, batch_size=batch_size, label=self.label)
        self.val_ds = self.df_to_dataset(self.test_df, batch_size=batch_size, label=self.label)

    def fit(self, model, metric, batch_size=256, learning_rate=0.1, callbacks = [], save_best=False):
        super().prepare_data(batch_size)
        # run native fit
        lr_schedule = schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=10000,
            decay_rate=0.95)
        if not callbacks:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=7, min_delta=1e-2),
                tf.keras.callbacks.ReduceLROnPlateau(patience=3),
                tf.keras.callbacks.LearningRateScheduler(schedule=lr_schedule),
                tf.keras.callbacks.TerminateOnNaN(),
                ]
        if save_best:
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath='/saved_models',
                    save_weights_only=True,
                    monitor='val_loss',
                    mode='auto',
                    verbose=0,
                    save_best_only=True)
            )
        if self.save_tf_logs:
            log_dir = "logs/fit/"
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
            callbacks += [tensorboard_callback]

        # we will stop after a 1/3rd of total epochs to determine whether this model is performing so poorly
        # as to justify an early abort
        history = None
        for epochs in [self.sub_epochs(1/3), self.sub_epochs(2/3)]:
            history = model.fit(
                self.train_ds, 
                validation_data=self.val_ds, 
                epochs=epochs, 
                batch_size=self.batch_size, 
                verbose=1, 
                callbacks=[callbacks]
            )
            score = history.history['val_mean_absolute_percentage_error'][-1]
            self.early_performance += [score]
            if len(self.early_performance) > 10:
                benchmark = moving_average(self.early_performance)
                if score > benchmark:
                    add_to_log("Performing poorly, discontinuing")
                    break                
        # return metrics
        return self.history.history[metric][-1]

    @staticmethod
    def df_to_dataset(dataframe:DataFrame, batch_size:int, shuffle=True, label=None):
        """ A utility method to create a tf.data dataset from a Pandas Dataframe """
        dataframe = dataframe.copy()
        if label:
            labels = dataframe.pop(label)
            ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        else:
            ds = tf.data.Dataset.from_tensor_slices((dict(dataframe)))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)
        ds = ds.prefetch(batch_size)
        return ds        

class SklearnAdaptor(AbstractAdaptor):
    def convert_pandas(self):
        self.X_train = self.train_df.drop(self.y_col_name, axis=1)
        self.y_train = self.train_df[self.y_col_name].copy()
        self.X_val = self.val_df.drop(self.y_col_name, axis=1)
        self.y_val = self.val_df[self.y_col_name].copy()

    def fit(self, model, metric, learning_rate, save_best=False):
        # we will stop after a 1/3rd of total epochs to determine whether this model is performing so poorly
        # as to justify an early abort
        history = score = None
        for epochs in [self.sub_epochs(1/3), self.sub_epochs(2/3)]:
            model.fit(
                self.X_train,
                self.y_train,
                max_iter = epochs,
                learning_rate = self.learning_rate,
            ) 
            score = None
        return score 
