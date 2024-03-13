import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate, Conv3DTranspose,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Activation

import numpy as np
import sys
sys.stdout = open('test_output.txt','w')

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class ReflectionPadding3D(Layer):
    def __init__(self, padding=(1,1,1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding3D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3] + 2 * self.padding[2], s[4])

    def call(self, x):
        padding_width = [[0, 0], [self.padding[0], self.padding[0]], [self.padding[1], self.padding[1]], [self.padding[2], self.padding[2]], [0, 0]]
        return tf.pad(x, padding_width, mode='REFLECT')

    def get_config(self):
        config = super(ReflectionPadding3D, self).get_config()
        config.update({
            'padding': self.padding
        })
        return config

def conv_block(input_tensor, num_filters, kernel_size, padding=(1,1,1),batch_norm=True, final_activation='relu'):
    x = ReflectionPadding3D(padding=padding)(input_tensor)
    x = Conv3D(num_filters, kernel_size, padding='valid')(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation(final_activation)(x)
    return x

def encoder_block(input_tensor, num_filters):
    encoder = conv_block(input_tensor, num_filters, kernel_size=(5, 5, 5),padding=(2,2,2), batch_norm=(num_filters != 32))
    pool = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(encoder)
    return pool, encoder

def decoder_block(input_tensor, concat_tensor, num_filters):
    decoder = UpSampling3D((2, 2, 2))(input_tensor)
    decoder = concatenate([concat_tensor, decoder], axis=-1)
    decoder = conv_block(decoder, num_filters, kernel_size=(3, 3, 3), padding=(1,1,1),final_activation='relu')
    return decoder

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

    outputs = Conv3D(1, (1, 1, 1), activation='tanh')(decoder1)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model

input_shape = (128, 128, 128, 1)

model = build_unet(input_shape)
model.compile(optimizer=Adam(), loss=losses.MeanSquaredError(), metrics=[tf.keras.metrics.RootMeanSquaredError()])
model.summary()

# Build model
#strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0","/gpu:3"])
#print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
train_dataset = tf.data.Dataset.from_tensor_slices(((np.load('/home/scratch/SDCunist/3d_DL/Train_sets/train_x.npy')),np.load('/home/scratch/SDCunist/3d_DL/Train_sets/train_y.npy'))).batch(4).prefetch(tf.data.experimental.AUTOTUNE)
#test_dataset = tf.data.Dataset.from_tensor_slices(((np.load('/home/scratch/SDCunist/3d_DL/trainset_1/test_x.npy')), np.load('/home/scratch/SDCunist/3d_DL/trainset_1/test_y.npy'))).batch(4).prefetch(tf.data.experimental.AUTOTUNE)

checkpoint_filepath="/home/scratch/SDCunist/3d_DL"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                               save_weights_only=False,
                                                               monitor='val_root_mean_squared_error',
                                                               mode='min',
                                                               save_best_only=True)
#with strategy.scope():
model.fit(train_dataset,epochs=100,validation_split=0.2,callbacks=[model_checkpoint_callback])
model.save("/home/scratch/SDCunist/3d_DL/copy1.h5")
