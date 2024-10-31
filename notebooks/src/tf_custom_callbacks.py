import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import layers, models
from tqdm import tqdm, tqdm_notebook, notebook
import numpy as np

# Custom callback to stop training based on F-score and accuracy
class tf_early_stopping(Callback):
    def __init__(self, patience=10, min_delta=0.01):
        super(tf_early_stopping, self).__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.best_weights = None
        self.best_fscore = 0
        self.best_accuracy = 0

    def on_epoch_end(self, epoch, logs=None):
        current_fscore = logs.get('val_f1_score')
        current_accuracy = logs.get('val_accuracy')

        # average f1 score
        current_fscore = np.mean(current_fscore)

        if current_fscore is None or current_accuracy is None:
            return
        
        if current_fscore >= self.best_fscore and current_accuracy >= self.best_accuracy:
                self.best_fscore = current_fscore
                self.best_accuracy = current_accuracy
                self.best_weights = self.model.get_weights()

        if (abs(current_fscore - self.best_fscore) < self.min_delta and
                abs(current_accuracy - self.best_accuracy) < self.min_delta):
            self.wait += 1
            if self.wait >= self.patience:
                print(f"Epoch {epoch}: Early stopping")
                self.model.stop_training = True
        else:
            self.wait = 0

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            print("Restoring model weights from the end of the best epoch.")

class tf_early_stopping_2(Callback):
    def __init__(self, patience=10, min_delta=0.01):
        super(tf_early_stopping_2, self).__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.best_weights = None
        self.best_fscore = 0
        self.best_accuracy = 0

    def on_epoch_end(self, epoch, logs=None):
        current_accuracy = logs.get('val_accuracy')

        if current_accuracy is None:
            return
        
        if current_accuracy >= self.best_accuracy:
                self.best_accuracy = current_accuracy
                self.best_weights = self.model.get_weights()

        if abs(current_accuracy - self.best_accuracy) < self.min_delta:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"Epoch {epoch}: Early stopping")
                self.model.stop_training = True
        else:
            self.wait = 0

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            print("Restoring model weights from the end of the best epoch.")

class TQDMProgressBar(Callback):
    def set_custom_name(self, name):
        self.bar_name = name

    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        self.steps = self.params['steps']

        # color_start = '\033[92m'  # Green text
        # color_end = '\033[0m'     # Reset to default color
        # bar_format = f'{color_start}{{l_bar}}{{bar:30}}{{r_bar}}{color_end}'

        if 'ipykernel' in sys.modules:
            self.progress_bar = notebook.tqdm(total=self.epochs * self.steps, desc=f'{self.bar_name} Training Progress', dynamic_ncols=True, leave=False)
        else:
            self.progress_bar = tqdm(total=self.epochs * self.steps, desc=f'{self.bar_name} Training Progress', dynamic_ncols=True, leave=False)

    def on_batch_end(self, batch, logs=None):
        self.progress_bar.update(1)
        self.progress_bar.set_postfix(logs, refresh=False)

    def on_train_end(self, logs=None):
        if self.model.stop_training:
            self.progress_bar.update(self.epochs * self.steps - self.progress_bar.n)
        self.progress_bar.close()

class tf_progress_bar_updater(Callback):
    def __init__(self, progress_bar, subcount=1):
        super(tf_progress_bar_updater, self).__init__()
        self.pbar = progress_bar
        self.epoch_done = 0
        self.subcount = subcount

    def on_epoch_end(self, epoch, logs=None):
        # logs = logs or {}
        self.pbar.update(1)
        self.epoch_done += 1
        # self.pbar.set_postfix(logs, refresh=False)
    
    def on_train_end(self, logs=None):
        if self.model.stop_training:
            self.pbar.update(self.subcount - self.epoch_done)