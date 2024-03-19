import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate, Conv3DTranspose,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Activation
import matplotlib.pyplot as plt
import numpy as np
import sys

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# Load the NumPy array
a = np.load("/home/scratch/SDCunist/3d_DL/trainset/SDC3_REAL.npy")[:10]

# Ensure array shape is compatible with model input
# (Add any necessary preprocessing or reshaping here)
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

# Load the model
model_path = "/home/scratch/SDCunist/3d_DL/copy1.h5"
try:
    loaded_model = load_model(model_path,custom_objects={'ReflectionPadding3D': ReflectionPadding3D})
except OSError as e:
    print(f"Error loading model: {e}")
    exit(1)

# Make predictions (modify batch size as needed)
# Ensure the batch size matches your model's requirements and GPU capabilities
try:
    prediction = loaded_model(a).numpy()  # Adjust the slice as per your batch size
except Exception as e:
    print(f"Error during prediction: {e}")
    exit(1)
print(prediction[5])
plt.imshow(prediction[5,:,:,10])
plt.savefig("ss")
# Save predictions
output_path = "/home/scratch/SDCunist/3d_DL/SDC3_test.npy"
try:
    np.save(output_path, prediction)
except Exception as e:
    print(f"Error saving predictions: {e}")
    exit(1)
