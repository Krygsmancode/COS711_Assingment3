import tensorflow as tf
from tensorflow.keras import layers, models

# Define the ResNet backbone (using ResNet50 as an example)
def resnet_backbone(input_shape, trainable=True):
    base_model = tf.keras.applications.ResNet50(
        input_shape=input_shape,
        include_top=False,
        # weights='imagenet',
    )
    base_model.trainable = trainable  # Freeze the layers for transfer learning
    return base_model

# Define SSD detection head
def ssd_head(num_classes, input_shape):
    # Layers to predict bounding boxes and class probabilities
    def ssd_block(x, filters, name):
        x = layers.Conv2D(filters, (3, 3), padding='same', name=name + '_conv')(x)
        x = layers.BatchNormalization(name=name + '_bn')(x)
        x = layers.ReLU(name=name + '_relu')(x)
        return x

    input_layer = layers.Input(shape=input_shape, name="input")
    backbone = resnet_backbone(input_shape)

    # Feature map from different layers of ResNet
    x = backbone(input_layer)
    x = ssd_block(x, 512, 'ssd_block1')
    x = ssd_block(x, 256, 'ssd_block2')

    # Bounding box predictions (location)
    locs = layers.Conv2D(4 * 4, (3, 3), padding='same', activation='linear', name="loc_conv")(x)  # 4 coordinates per box
    locs = layers.Reshape((-1, 4), name="loc_reshape")(locs)

    # Class predictions
    confs = layers.Conv2D(num_classes * 4, (3, 3), padding='same', activation='softmax', name="conf_conv")(x)  # num_classes per box
    confs = layers.Reshape((-1, num_classes), name="conf_reshape")(confs)

    # Concatenate location and confidence predictions
    output = layers.Concatenate(axis=-1, name="detection_output")([locs, confs])

    # model = models.Model(inputs=input_layer, outputs=(backbone.output, output))
    model = models.Model(inputs=input_layer, outputs=output)
    # model = models.Model(inputs=input_layer, outputs=(locs, confs))
    return model

# Define the SSD model
def ssd_resnet(num_classes, input_shape):
    # input_shape = (300, 300, 3)  # Input image size
    model = ssd_head(num_classes, input_shape)
    return model

def create_model(output_shape, input_shape):
    model = ssd_resnet(output_shape, input_shape)
    return model