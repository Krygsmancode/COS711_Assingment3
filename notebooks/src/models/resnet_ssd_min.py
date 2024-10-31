import tensorflow as tf
from tensorflow.keras import layers, models

# Layers to predict bounding boxes and class probabilities
def ssd_block(x, filters, name):
    x = layers.Conv2D(filters, (3, 3), padding='same', name=name + '_conv')(x)
    x = layers.BatchNormalization(name=name + '_bn')(x)
    x = layers.ReLU(name=name + '_relu')(x)
    return x

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

# Define a basic residual block with configurable shortcut kernel and stride
def basic_residual_block2(x, filters, kernel_size=(3, 3), stride=1, short_conv_kernel=(1, 1), short_conv_stride=1, name="res_block"):
    shortcut = x
    
    # First convolution
    x = layers.Conv2D(filters, kernel_size, padding='same', strides=stride, name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.ReLU(name=f"{name}_relu1")(x)

    # Second convolution
    x = layers.Conv2D(filters, kernel_size, padding='same', name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)

    # Adjust shortcut to match output shape if necessary
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, short_conv_kernel, strides=short_conv_stride, name=f"{name}_shortcut")(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_shortcut_bn")(shortcut)

    # Add and activate
    x = layers.Add(name=f"{name}_add")([x, shortcut])
    x = layers.ReLU(name=f"{name}_out")(x)
    return x

def basic_convolutional_block(x, filters, kernel_size=(3, 3), stride=1, name="conv_block"):
    return basic_residual_block2(x, filters, kernel_size=kernel_size, stride=stride, short_conv_kernel=(1, 1), short_conv_stride=2, name=name)

def basic_identity_block(x, filters, kernel_size=(3, 3), name="identity_block"):
    return basic_residual_block2(x, filters, kernel_size=kernel_size, stride=1, short_conv_kernel=(1, 1), short_conv_stride=1, name=name)


def small_resnet_ssd(input_shape=(128, 128, 3), num_classes=10, num_boxes=50):
    inputs = layers.Input(shape=input_shape)
    
    # Initial Conv Layer
    x = layers.Conv2D(16, (7, 7), strides=2, padding='same', name="initial_conv")(inputs)
    x = layers.BatchNormalization(name="initial_bn")(x)
    x = layers.ReLU(name="initial_relu")(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', name="initial_pool")(x)
    
    # Small ResNet Backbone
    x = basic_convolutional_block(x, filters=16, name="conv_block1")
    x = basic_identity_block(x, filters=16, name="identity_block1")
    x = basic_convolutional_block(x, filters=32, stride=2, name="conv_block2")
    x = basic_identity_block(x, filters=32, name="identity_block2")
    x = basic_convolutional_block(x, filters=64, stride=2, name="conv_block3")
    x = basic_identity_block(x, filters=64, name="identity_block3")
    
    # SSD Prediction Heads at different scales
    # First SSD Block
    x = ssd_block(x, filters=64, name="ssd1")
    x = ssd_block(x, filters=128, name="ssd2")

    # SSD Output
    locs = layers.Conv2D(4 * 4, (3, 3), padding='same', name="ssd_locs")(x)
    locs = layers.Reshape((-1, 4), name="ssd_locs_reshape")(locs)

    confs = layers.Conv2D(4 * num_classes, (3, 3), padding='same', name="ssd_confs")(x)
    confs = layers.Reshape((-1, num_classes), name="ssd_confs_reshape")(confs)

    loc_conf = layers.Concatenate(axis=2, name="ssd_output")([locs, confs])

    # loc_conf_shape = tf.shape(loc_conf)
    # print("loc_conf shape:", loc_conf_shape)
    # num_out_boxes = loc_conf_shape[1]
    # print("num_out_boxes:", num_out_boxes)
    # pool_size = num_out_boxes // num_boxes

    # shrink the output to the number of boxes (max pool)
    outputs = layers.MaxPool1D(pool_size=10, name="ssd_output_shrink")(loc_conf)
    # Resize the output to the exact number of boxes
    # final_output_size = num_boxes * (4 + num_classes)
    # outputs = layers.Reshape((num_boxes, 4 + num_classes), name="ssd_final_output")(loc_conf[:, :final_output_size])

    # # Ensure loc_conf has exactly num_boxes rows
    # def pad_or_truncate(tensor, target_size):
    #     tensor_shape = tf.shape(tensor)
    #     if tensor_shape[1] > target_size:
    #         # Slice if too large
    #         return tensor[:, :target_size, :]
    #     elif tensor_shape[1] < target_size:
    #         # Pad if too small
    #         padding = tf.constant([[0, 0], [0, target_size - tensor_shape[1]], [0, 0]])
    #         return tf.pad(tensor, padding)
    #     else:
    #         return tensor

    # loc_conf = layers.Lambda(lambda x: pad_or_truncate(x, num_boxes), name="ssd_pad_truncate")(loc_conf)

    # outputs = loc_conf



    # ssd1_output = layers.Conv2D(num_boxes * (4 + num_classes), (1, 1), padding='same', name="ssd1_output")(ssd1)
    
    # # Second SSD Block
    # ssd2 = layers.Conv2D(128, (3, 3), strides=2, padding='same', name="ssd2_conv")(x)
    # ssd2 = layers.BatchNormalization(name="ssd2_bn")(ssd2)
    # ssd2 = layers.ReLU(name="ssd2_relu")(ssd2)
    # ssd2_output = layers.Conv2D(num_boxes * (4 + num_classes), (1, 1), padding='same', name="ssd2_output")(ssd2)

    # # Flatten and Concatenate SSD Outputs
    # ssd1_flat = layers.Reshape((-1, 4 + num_classes), name="ssd1_flat")(ssd1_output)
    # ssd2_flat = layers.Reshape((-1, 4 + num_classes), name="ssd2_flat")(ssd2_output)
    # outputs = layers.Concatenate(axis=1, name="ssd_concat")([ssd1_flat, ssd2_flat])

    model = models.Model(inputs, outputs, name="small_resnet_ssd")
    return model

def small_resnet_ssd_2x(input_shape=(128, 128, 3), num_classes=10, num_boxes=50):
    inputs = layers.Input(shape=input_shape)
    
    # Initial Conv Layer
    x = layers.Conv2D(16, (7, 7), strides=2, padding='same', name="initial_conv")(inputs)
    x = layers.BatchNormalization(name="initial_bn")(x)
    x = layers.ReLU(name="initial_relu")(x)
    x_initial = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', name="initial_pool")(x)
    
    # Small ResNet Backbone Kernals (3, 3) Parallel Block set 1
    x = basic_identity_block(x, filters=16, name="identity_block1")
    x = basic_convolutional_block(x, filters=32, stride=2, name="conv_block2")
    x = basic_identity_block(x, filters=32, name="identity_block2")
    x = basic_convolutional_block(x, filters=64, stride=2, name="conv_block3")
    x = basic_identity_block(x, filters=64, name="identity_block3")

    # Small ResNet Backbone Kernals (5, 5) Parallel Block set 2
    x2 = basic_identity_block(x_initial, filters=16, kernel_size=(5, 5), name="identity_block1_2")
    x2 = basic_convolutional_block(x2, filters=32, kernel_size=(5, 5), stride=2, name="conv_block2_2")
    x2 = basic_identity_block(x2, filters=32, kernel_size=(5, 5), name="identity_block2_2")
    x2 = basic_convolutional_block(x2, filters=64, kernel_size=(5, 5), stride=2, name="conv_block3_2")
    x2 = basic_identity_block(x2, filters=64, kernel_size=(5, 5), name="identity_block3_2")

    # small ResNet Backbone Kernals (7, 7) Parallel Block set 3
    x3 = basic_identity_block(x_initial, filters=16, kernel_size=(7, 7), name="identity_block1_3")
    x3 = basic_convolutional_block(x3, filters=32, kernel_size=(7, 7), stride=2, name="conv_block2_3")
    x3 = basic_identity_block(x3, filters=32, kernel_size=(7, 7), name="identity_block2_3")
    x3 = basic_convolutional_block(x3, filters=64, kernel_size=(7, 7), stride=2, name="conv_block3_3")
    x3 = basic_identity_block(x3, filters=64, kernel_size=(7, 7), name="identity_block3_3")

    # reshape the parallel blocks
    x = layers.Reshape((-1, 64), name="reshape_block1")(x)
    x2 = layers.Reshape((-1, 64), name="reshape_block2")(x2)
    x3 = layers.Reshape((-1, 64), name="reshape_block3")(x3)

    # Concatenate the parallel blocks
    x = layers.Concatenate(axis=3, name="concatenate_parallel_blocks")([x, x2, x3])
    
    # SSD Prediction Heads at different scales
    # First SSD Block
    x = ssd_block(x, filters=64, name="ssd1")
    x = ssd_block(x, filters=128, name="ssd2")

    # SSD Output
    locs = layers.Conv2D(4 * 4, (3, 3), padding='same', name="ssd_locs")(x)
    locs = layers.Reshape((-1, 4), name="ssd_locs_reshape")(locs)

    confs = layers.Conv2D(4 * num_classes, (3, 3), padding='same', name="ssd_confs")(x)
    confs = layers.Reshape((-1, num_classes), name="ssd_confs_reshape")(confs)

    loc_conf = layers.Concatenate(axis=2, name="ssd_output")([locs, confs])

    outputs = layers.MaxPool1D(pool_size=10, name="ssd_output_shrink")(loc_conf)

    model = models.Model(inputs, outputs, name="small_resnet_ssd")
    return model
