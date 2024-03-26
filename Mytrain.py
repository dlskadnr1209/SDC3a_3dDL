import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate, Conv3DTranspose,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Activation
from tensorflow.distribute import MirroredStrategy
import os
import numpy as np
import sys

# Custom 3D Reflection Padding Layer
class ReflectionPadding3D(Layer):
    def __init__(self, padding=(1,1,1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding3D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        # Calculate and return the output shape
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3] + 2 * self.padding[2], s[4])

    def call(self, x):
        # Perform the actual padding operation
        padding_width = [[0, 0], [self.padding[0], self.padding[0]], [self.padding[1], self.padding[1]], [self.padding[2], self.padding[2]], [0, 0]]
        return tf.pad(x, padding_width, mode='REFLECT')

    def get_config(self):
        # Necessary for saving and loading the model with custom layer
        config = super(ReflectionPadding3D, self).get_config()
        config.update({
            'padding': self.padding
        })
        return config

# Convolution block used in the U-Net
def conv_block(input_tensor, num_filters, kernel_size, padding=(1,1,1),batch_norm=True, final_activation='relu'):
    # Apply reflection padding
    x = ReflectionPadding3D(padding=padding)(input_tensor)
    # Perform convolution
    x = Conv3D(num_filters, kernel_size, padding='valid')(x)
    # Apply batch normalization if required    
    if batch_norm:
        x = BatchNormalization()(x)
    # Apply batch normalization if required
    x = Activation(final_activation)(x)
    return x

# Encoder block in U-Net architecture
def encoder_block(input_tensor, num_filters):
    # Convolution followed by pooling
    encoder = conv_block(input_tensor, num_filters, kernel_size=(5, 5, 5),padding=(2,2,2), batch_norm=(num_filters != 32))
    pool = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(encoder)
    return pool, encoder

# Decoder block in U-Net architecture
def decoder_block(input_tensor, concat_tensor, num_filters):
    # Upsampling followed by concatenation and convolution
    decoder = UpSampling3D((2, 2, 2))(input_tensor)
    decoder = concatenate([concat_tensor, decoder], axis=-1)
    decoder = conv_block(decoder, num_filters, kernel_size=(3, 3, 3), padding=(1,1,1),final_activation='relu')
    return decoder

# Function to build the U-Net model
def build_unet(input_shape):
    inputs = Input(input_shape)
    # Encoder
    encoder1, conv1 = encoder_block(inputs, 32)
    encoder2, conv2 = encoder_block(encoder1, 64)
    encoder3, conv3 = encoder_block(encoder2, 128)
    encoder4, conv4 = encoder_block(encoder3, 256)
    encoder5, conv5 = encoder_block(encoder4,512)

    # Bottleneck
    bottleneck = conv_block(encoder5, 1024, kernel_size=(3, 3, 3))

    # Decoder
    decoder5 = decoder_block(bottleneck, conv5, 512)
    decoder4 = decoder_block(decoder5, conv4, 256)
    decoder3 = decoder_block(decoder4, conv3, 128)
    decoder2 = decoder_block(decoder3, conv2, 64)
    decoder1 = decoder_block(decoder2, conv1, 32)

    # Final convolution
    outputs = Conv3D(1, (1, 1, 1), activation='tanh')(decoder1)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model

# Function to train the U-Net model
def training(input_path, output_path, epochs, validation_ratio, batch_size):

    global numgpus

    strategy = MirroredStrategy()
    strategy.extended._cross_device_ops = tf.distribute.ReductionToOneDevice() # if NCCL do not work for your setting, please try this

    with strategy.scope():
        input_shape = (128, 128, 128, 1)
        model = build_unet(input_shape)
        model.compile(optimizer=Adam(), loss=losses.MeanSquaredError(), metrics=[tf.keras.metrics.RootMeanSquaredError()])

    model.summary()

    # Load the data
    x = np.load(os.path.join(input_path, 'train_x.npy'))
    y = np.load(os.path.join(input_path, 'train_y.npy'))

    # Calculate the number of training samples
    num_train_samples = int((1 - validation_ratio) * x.shape[0])

    # Split the data
    x_train, x_val = x[:num_train_samples], x[num_train_samples:]
    y_train, y_val = y[:num_train_samples], y[num_train_samples:]

    if numgpus==0:
        global_batch_size=batch_size*numgpus:
    else:
        global_batch_size=batch_size*numgpus

    # Create tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(num_train_samples).batch(global_batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(global_batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    # Checkpoint callback
    checkpoint_filepath = os.path.join(output_path, 'checkpoint')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                                   save_weights_only=False,
                                                                   monitor='val_root_mean_squared_error',
                                                                   mode='min',
                                                                   save_best_only=True)

    # Train the model
    model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=[model_checkpoint_callback])

    # Save the final model
    model.save(os.path.join(output_path, '3Dmodel.h5'))
