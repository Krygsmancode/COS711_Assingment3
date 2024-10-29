import tensorflow as tf
from tensorflow.keras import layers, models

# Define the ResNet backbone (using ResNet50 as an example)
def resnet_backbone(input_shape, trainable=True, imagenet_weights=True):
    weights = None
    if not trainable and imagenet_weights:
        print("Using ResNet50 as a feature extractor with imagenet weights")
        weights = 'imagenet'
    elif not trainable and not imagenet_weights:
        print("Using ResNet50 as a feature extractor")
    elif trainable and imagenet_weights:
        print("Using ResNet50 as a feature extractor with imagenet weights and fine-tuning")
        weights = 'imagenet'
    elif trainable and not imagenet_weights:
        print("Using ResNet50 as a feature extractor with fine-tuning")
    else:
        raise ValueError("Invalid combination of arguments")

    base_model = tf.keras.applications.ResNet50(
        input_shape=input_shape,
        include_top=False,
        weights=weights
    )

    base_model.trainable = trainable  # Freeze the layers for transfer learning
    return base_model

# Layers to predict bounding boxes and class probabilities
def ssd_block(x, filters, name):
    x = layers.Conv2D(filters, (3, 3), padding='same', name=name + '_conv')(x)
    x = layers.BatchNormalization(name=name + '_bn')(x)
    x = layers.ReLU(name=name + '_relu')(x)
    return x

# Define SSD detection head
def ssd_head(num_classes, input_shape, backbone=None):

    input_layer = layers.Input(shape=input_shape, name="input")
    if backbone is None:
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
def create_model(output_shape, input_shape):
    model = ssd_head(output_shape, input_shape)
    return model

# Define a basic residual block
def basic_residual_block(x, filters, kernel_size=(3, 3), stride=1, name="res_block"):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, padding='same', strides=stride, name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.ReLU(name=f"{name}_relu1")(x)

    x = layers.Conv2D(filters, kernel_size, padding='same', name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)

    # Adjust shortcut to match output shape if necessary
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), strides=stride, name=f"{name}_shortcut")(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_shortcut_bn")(shortcut)
    
    x = layers.Add(name=f"{name}_add")([x, shortcut])
    x = layers.ReLU(name=f"{name}_out")(x)
    return x

# Define customizable ResNet-like backbone
def custom_resnet_backbone(input_shape, num_blocks=3, initial_filters=32, growth_rate=2, trainable=True):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(initial_filters, (3, 3), padding='same', strides=2, name="initial_conv")(inputs)
    x = layers.BatchNormalization(name="initial_bn")(x)
    x = layers.ReLU(name="initial_relu")(x)

    filters = initial_filters
    # Add configurable number of residual blocks
    for block_idx in range(num_blocks):
        x = basic_residual_block(x, filters, stride=1 if block_idx == 0 else 2, name=f"block{block_idx + 1}")
        filters *= growth_rate  # Increase filters after each block for a progressive architecture

    model = models.Model(inputs, x, name="custom_resnet_backbone")
    model.trainable = trainable
    return model

# SSD model with configurable SSD blocks after the backbone
def ssd_with_min_resnet_backbone(input_shape, num_classes, num_res_blocks=3, initial_filters=32, growth_rate=2, num_ssd_blocks=2, shrink_rate=2, trainable_backbone=True):
    # Initialize backbone
    backbone = custom_resnet_backbone(input_shape, num_blocks=num_res_blocks, initial_filters=initial_filters, 
                                      growth_rate=growth_rate, trainable=trainable_backbone)
    
    x = backbone.output
    filters = initial_filters * growth_rate ** num_res_blocks  # Starting filter size for SSD blocks
    
    # Add configurable number of SSD blocks after the backbone based on the shrink rate
    for block_idx in range(num_ssd_blocks):
        x = ssd_block(x, filters, f"ssd_block{block_idx + 1}")
        filters //= shrink_rate  # Decrease filters after each block for a progressive architecture
    

    # Bounding box predictions (location)
    locs = layers.Conv2D(4 * 4, (3, 3), padding='same', activation='linear', name="loc_conv")(x)
    locs = layers.Reshape((-1, 4), name="loc_reshape")(locs)

    # Class predictions
    confs = layers.Conv2D(num_classes * 4, (3, 3), padding='same', activation='softmax', name="conf_conv")(x)
    confs = layers.Reshape((-1, num_classes), name="conf_reshape")(confs)

    # Concatenate location and confidence predictions
    output = layers.Concatenate(axis=-1, name="detection_output")([locs, confs])

    model = models.Model(inputs=backbone.input, outputs=output, name="ssd_with_custom_backbone")
    return model

# # Define SSD head with custom backbone
# def ssd_with_custom_backbone(input_shape, num_classes, num_blocks=3, initial_filters=32, growth_rate=2, trainable_backbone=True):
#     backbone = custom_resnet_backbone(input_shape, num_blocks=num_blocks, initial_filters=initial_filters, 
#                                       growth_rate=growth_rate, trainable=trainable_backbone)
    
#     x = backbone.output
#     x = layers.Conv2D(256, (3, 3), padding='same', activation='relu', name="ssd_block")(x)
    
#     locs = layers.Conv2D(4 * 4, (3, 3), padding='same', activation='linear', name="loc_conv")(x)
#     locs = layers.Reshape((-1, 4), name="loc_reshape")(locs)

#     confs = layers.Conv2D(num_classes * 4, (3, 3), padding='same', activation='softmax', name="conf_conv")(x)
#     confs = layers.Reshape((-1, num_classes), name="conf_reshape")(confs)

#     output = layers.Concatenate(axis=-1, name="detection_output")([locs, confs])

#     model = models.Model(inputs=backbone.input, outputs=output, name="ssd_with_custom_backbone")
#     return model