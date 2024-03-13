import argparse
import Mytrainset
import Mytest
import Mytrain
import tensorflow as tf

parser = argparse.ArgumentParser(description='Point Source Removal Deep Learning')

#Required
parser.add_argument('input_path', type=str, help='Input Path')
parser.add_argument('output_path', type=str, help='Output Path')

#TRAIN-SET
parser.add_argument('--making_train_set', type=bool, default=False, help='If you want to make training set : True')
parser.add_argument('--number_of_trainset', type=int, default=2560, help='Number of Training sets')
parser.add_argument('--agumentation', type=bool, default=True, help='Rotational Agumentation?(Number of Training sets x 4) : True')
parser.add_argument('--background', type=bool, default=False, help='If you want to make background cube : True')
parser.add_argument('--making_test_data', type=bool, default=False, help='If you want to make SDC3 testdata : True')
parser.add_argument('--frequency_channels',type=int,default=264,help='How many channels are you goint to use? / 1channel : 0.1MHz')

#TRAINING
parser.add_argument('--epochs', type=int, default=10, help='Epoch')
parser.add_argument('--validation_split', type=float, default=0.2, help='Validation Split')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')

#TEST
parser.add_argument('--apply_to_SDC3a', type=bool, default=True, help='To apply model to SDC3 : True/False')

if __name == '__main__':
    args=parser.parse_args()

    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    Mytrainset.process_data(args.output_path,args.number_of_trainset,args.)
    Mytrain.training(args.input_path,args.output_path,args.epochs,args.validation_split,args.batch_size)
    Mytest.test()
