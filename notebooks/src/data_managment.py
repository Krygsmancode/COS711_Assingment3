import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

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
        
    def load_data(self, alt_image_dir=None, expected_image_shape=None, resize_images=False):
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
            image = None
            try:
                image = plt.imread(f"{self.image_dir}/{name}") # load the image with format of (height, width, channels) using

            except FileNotFoundError:
                if alt_image_dir:
                    image = plt.imread(f"{alt_image_dir}/{name}")

                else:
                    print(f"Image {name} not found in {self.image_dir} or {alt_image_dir}")
                    continue
            
            # print(f"Loaded image {name} with shape {image.shape}")

            if expected_image_shape: # check if the image has the expected shape
                if image.shape != expected_image_shape:
                    print(f"Image {name} has an unexpected shape:", end=' ')
                    
                    if resize_images:
                        old_shape = image.shape
                        # image = tf.image.resize(image, expected_image_shape[:2])

                        # resize the image to the expected shape
                        # Resize image
                        output_size = expected_image_shape[:2][::-1]
                        img_resized = cv2.resize(image, output_size)
                        image = img_resized

                        print(f"Resized from {old_shape} to {image.shape}")
                    else:
                        print(f"{image.shape} instead of {expected_image_shape}")
                        continue

            X.append(image)

        X = np.array(X)

        # y data format: (ymin,xmin,ymax,xmax,class1,class2,class3,...)

        # remove the 'Image_ID' column and the 'confidence' column
        y = data.drop(['Image_ID', 'confidence'], axis=1)

        # one hot encode the class
        num_classes = len(np.unique(y['class']))
        # pandas get_dummies is equivalent to one hot encoding
        y_class = pd.get_dummies(y['class'])

        # concatenate the class one hot encoding with the bounding box
        y = pd.concat([y.drop('class', axis=1), y_class], axis=1)

        # print y headings
        print(f"y headings: {y.columns}")

        return X, y
    
    def checkShapes(self, X, y):
        print(f"X shape: (Number of images, Image Height, Image Width, Image Channels): {X.shape}")
        print(f"y shape: (Number of images, 4 + Number of Classes): {y.shape}")