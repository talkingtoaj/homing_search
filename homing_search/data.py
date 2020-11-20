from sklearn.model_selection import train_test_split
from pandas import DataFrame
import tensorflow as tf
from tensorflow.keras.optimizers import schedules
from .utils import add_to_log, moving_average
from statistics import median

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
        # run native fit
        # return metrics
        raise NotImplemented

class KerasAdaptor(AbstractAdaptor):
    def __init__(self, data, label, save_tf_logs=False, searching_mode=True):
        """ searching_mode gives up more easily, set True if doing HomingSearch, set False if making model for production predictions """
        super().__init__(data, label)        
        self.save_tf_logs = save_tf_logs
        self.searching_mode = searching_mode
        self.early_performance = []
        
    def convert_pandas(self, batch_size):
        self.train_ds = self.df_to_dataset(self.train_df, batch_size=batch_size, label=self.label)
        self.val_ds = self.df_to_dataset(self.val_df, batch_size=batch_size, label=self.label)

    def fit(self, model, metric, epochs, batch_size=256, learning_rate=0.1, callbacks = [], save_best=False):
        super().prepare_data(batch_size)
        self.epochs = epochs
        # run native fit
        lr_schedule = schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=10000,
            decay_rate=0.95)
        if not callbacks:
            es_patience = 7 if self.searching_mode else 15
            red_patience = 3 if self.searching_mode else 6
            
            callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=es_patience, min_delta=1e-2),
                tf.keras.callbacks.ReduceLROnPlateau(patience=red_patience),
                tf.keras.callbacks.LearningRateScheduler(schedule=lr_schedule),
                tf.keras.callbacks.TerminateOnNaN(),
                ]
        if save_best:
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath='saved_models/s',
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
        for epochs in [round(epochs*.33), round(epochs*.66)]:
            history = model.fit(
                self.train_ds, 
                validation_data=self.val_ds, 
                epochs=epochs, 
                batch_size=batch_size, 
                verbose=1, 
                callbacks=[callbacks]
            )
            score = history.history['val_mean_absolute_percentage_error'][-1]
            self.early_performance += [score]
            if len(self.early_performance) > 10:
                benchmark = median(self.early_performance) # median ignore crazy excessive outliers
                if score > benchmark:
                    add_to_log("Performing poorly, discontinuing")
                    break                
        # return metrics
        self.model = model
        self.history = history
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
    # TODO: This is not complete
    
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
