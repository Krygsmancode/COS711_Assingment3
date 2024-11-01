import tensorflow as tf
from tensorflow.keras import layers, models

class model_builder(): # base abstract class
    def create_model(output_shape = 1):
        pass

    def visualize_model(model):
        pass

class model_builder_v0(model_builder):
    __doc__ = "This is the first version of the model builder"
    def create_model(output_shape = 1):
        input_shape = (30, 30, 1)
        # Define the input shape
        inputs = layers.Input(shape=input_shape)

        # Branch 1: l1a
        l1a = layers.Conv2D(5, (5, 5), strides=(1, 1), activation='relu')(inputs)

        # Branch 2: l1b
        l1b = layers.Conv2D(7, (3, 3), strides=(1, 1), activation='relu')(inputs)
        l1b2 = layers.Conv2D(1, (1, 1), strides=(1, 1), activation='relu')(l1b)  # Compress to 1 channel

        # From l1b2 to l2a
        l2a = layers.Conv2D(5, (3, 3), strides=(1, 1), activation='relu')(l1b2)

        # Max pooling from l2a to l2a2
        l2a2 = layers.MaxPooling2D((2, 2), strides=(2, 2))(l2a)

        # Dense layer from l2a to l3a
        l3a = layers.Flatten()(l2a2)
        l3a = layers.Dense(5, activation='relu')(l3a)

        # max pooling from l1a to l1a2
        l1a2 = layers.MaxPooling2D((2, 2), strides=(2, 2))(l1a)

        # Flatten l1a to l2b
        l2b = layers.Flatten()(l1a2)

        # Combine l3a and l2b into l4a
        l4a = layers.concatenate([l3a, l2b])

        # Final output layer (example: for binary classification)
        output = layers.Dense(output_shape, activation='softmax')(l4a)

        # Create the model
        model = models.Model(inputs=inputs, outputs=output)

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def visualize_model(model):
        # Visualize the model
        tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

        return None