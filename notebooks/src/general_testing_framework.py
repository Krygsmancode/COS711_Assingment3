import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class ModelTrainingFrameworkBasic:
    """
    A class to train a model using tensorflow and evaluate it.

    Attributes

    model: tf.keras.Model
        The model to be trained and evaluated.

    X_data: np.array
        The input data to be used for training.

    y_data: np.array
        The target data to be used for training.

    batch_size: int
        The batch size to be used during training.

    epochs: int
        The number of epochs to train the model for.

    test_size: float
        The proportion of the data to be used for validation.
    """
    def __init__(self, model, X_data, y_data, batch_size=32, epochs=20, test_size=0.2):
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.history = None
        self.X_data = X_data
        self.y_data = y_data
        self.test_size = test_size

    def prepare_dataset(self, output_boundig_size=1):
        # duplicate the data to match the output size shape=(num_data, 4+num_classes_one_hot) --> shape=(num_data, output_boundig_size, 4+num_classes_one_hot)
        new_Y_data = np.zeros((self.y_data.shape[0], output_boundig_size, self.y_data.shape[1]))
        for i in range(self.y_data.shape[0]):
            new_Y_data[i] = np.tile(self.y_data[i], (output_boundig_size, 1))
        return new_Y_data


    
    def train(self, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy', 'f1_score'], callbacks=None):
        
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # get the output size of the model
        output_boundig_size = self.model.output_shape[1]
        new_y_data = self.prepare_dataset(output_boundig_size)

        self.history = self.model.fit(self.X_data, new_y_data, batch_size=self.batch_size, epochs=self.epochs, validation_split=self.test_size, callbacks=callbacks)
        
        return self.history
    
class ModelTrainingFramework:
    """
    A class to train a model using tensorflow and evaluate it.

    Attributes

    model: tf.keras.Model
        The model to be trained and evaluated.

    train_data: tuple
        A tuple containing the training data in the form (X_train, y_train).

    val_data: tuple
        A tuple containing the validation data in the form (X_val, y_val).

    batch_size: int
        The batch size to be used during training.

    epochs: int
        The number of epochs to train the model for.
    """
    def __init__(self, model, train_data, val_data, batch_size=32, epochs=20):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        self.epochs = epochs 
        
    def prepare_dataset(self):
        train_dataset = tf.data.Dataset.from_tensor_slices(self.train_data)
        train_dataset = train_dataset.batch(self.batch_size).shuffle(buffer_size=1000)
        
        val_dataset = tf.data.Dataset.from_tensor_slices(self.val_data)
        val_dataset = val_dataset.batch(self.batch_size)
        
        return train_dataset, val_dataset
    
    def train(self):
        train_dataset, val_dataset = self.prepare_dataset()
        
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy', 'f1_score'])
        history = self.model.fit(train_dataset, epochs=self.epochs, validation_data=val_dataset)
        
        return history
    
    def plot_history(self, history):
        plt.figure(figsize=(12,4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.show()
        
    def evaluate_model(self):
        val_dataset = tf.data.Dataset.from_tensor_slices(self.val_data).batch(self.batch_size)
        loss, accuracy = self.model.evaluate(val_dataset)
        print(f"Validation Loss: {loss}")
        print(f"Validation Accuracy: {accuracy}")

class DataLoader:
    """
    A class to load data from a file and preprocess it.

    Attributes

    file_path: str
        The path to the file containing the csv data. (Image_ID,class,confidence,ymin,xmin,ymax,xmax,augmented)

    image_dir: str
        The directory containing the images.
    """
    def __init__(self, file_path, image_dir):
        self.file_path = file_path
        self.image_dir = image_dir
        
    def load_data(self):
        data = pd.read_csv(self.file_path)

        # remove the 'augmented' column
        data = data.drop('augmented', axis=1)

        # remove rows with NaN values
        data = data.dropna()

        # split x (load the images from id) and y (the class and bounding box)
        X_names = data['Image_ID']

        # load the images
        X = []
        for name in X_names:
            image = plt.imread(f"{self.image_dir}/{name}")
            X.append(image)

        X = np.array(X)

        # y data format: (ymin,xmin,ymax,xmax,class1,class2,class3,...)

        # remove the 'Image_ID' column and the 'confidence' column
        y = data.drop(['Image_ID', 'confidence'], axis=1)

        # one hot encode the class
        num_classes = len(np.unique(y['class']))
        y_class = tf.one_hot(y['class'], num_classes)

        # concatenate the class one hot encoding with the bounding box
        y = pd.concat([y.drop('class', axis=1), y_class], axis=1)

        return X, y
    
    def checkShapes(self, X, y):
        print(f"X shape: (Number of images, Image Height, Image Width, Image Channels): {X.shape}")
        print(f"y shape: (Number of images, 4 + Number of Classes): {y.shape}")

        

