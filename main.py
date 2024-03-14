import argparse
import Mytrainset
import Mytest
import Mytrain
import tensorflow as tf

parser = argparse.ArgumentParser(description='Point Source Removal Deep Learning')

# Required arguments
parser.add_argument('input_path', type=str, help='Input Path')
parser.add_argument('output_path', type=str, help='Output Path')

# TRAIN-SET arguments
parser.add_argument('--making_train_set', action='store_true', help='Make training set: True/False')
parser.add_argument('--number_of_trainset', type=int, default=2560, help='Number of Training sets')
parser.add_argument('--agumentation', action='store_true', help='Rotational Agumentation? (Number of Training sets x 4): True/False')
parser.add_argument('--background', action='store_true', help='Make background cube: True/False')
parser.add_argument('--making_test_data', action='store_true', help='Make SDC3 test data: True/False')
parser.add_argument('--frequency_channels', type=int, default=264, help='Number of frequency channels (1 channel = 0.1MHz)')

# TRAINING arguments
parser.add_argument('--epochs', type=int, default=10, help='Number of Epochs')
parser.add_argument('--validation_split', type=float, default=0.2, help='Validation Split')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size')

# TEST arguments
parser.add_argument('--apply_to_SDC3a', action='store_true', help='Apply model to SDC3: True/False')

if __name__ == '__main__':
    args = parser.parse_args()

    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if args.making_train_set:
        Mytrainset.process_data(args.output_path, args.number_of_trainset, args.agumentation)

    Mytrain.training(args.input_path, args.output_path, args.epochs, args.validation_split, args.batch_size)

    if args.apply_to_SDC3a:
        Mytest.test()
