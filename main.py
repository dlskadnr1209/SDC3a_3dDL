import argparse
import Mytrainset
import Mytest
import Mytrain
import tensorflow as tf

# Initialize argument parser
parser = argparse.ArgumentParser(description='Point Source Removal Deep Learning')

# Required arguments: input and output paths
parser.add_argument('input_path', type=str, help='Input Path')
parser.add_argument('output_path', type=str, help='Output Path')

# Optional arguments for training dataset configuration
parser.add_argument('--making_train_set', action='store_true', help='Make training set')
parser.add_argument('-n','--number_of_trainset', type=int, default=2560, help='Number of Training sets')
parser.add_argument('-a','--augmentation', action='store_true', help='Rotational Augmentation (Number of Training sets x 4)')
parser.add_argument('-b','--background', action='store_true', help='Make background cube')
parser.add_argument('--making_test_data', action='store_true', help='Make SDC3 test data')
parser.add_argument('-f','--frequency_channels', type=int, default=264, help='Number of frequency channels (1 channel = 0.1MHz)')

# Arguments for model training
parser.add_argument('-e','--epochs', type=int, default=10)
parser.add_argument('-v','--validation_split', type=float, default=0.2)
parser.add_argument('--batch_size', type=int, default=4)

# Argument for applying the model to test data
parser.add_argument('--apply_to_SDC3a', action='store_true', help='Apply model to SDC3')

# Parse arguments and start the process
if __name__ == '__main__':
    args = parser.parse_args()

    # Configure GPU settings for TensorFlow
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    global numgpus
    numgpus=len(gpus)

    # Process for generating training data
    if args.making_train_set:
        Mytrainset.process_data(args.output_path, args.number_of_trainset, args.agumentation)

    # Training the model
    Mytrain.training(args.input_path, args.output_path, args.epochs, args.validation_split, args.batch_size)

    # Applying the trained model to SDC3 test data
    if args.apply_to_SDC3a:
        Mytest.test()
