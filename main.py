import argparse
import Mytrainset
import Mytest
import Mytrain

#
parser = argparse.ArgumentParser(description='Point Source Removal Deep Learning')

#
parser.add_argument('input_path', type=str, default='./', help='Input Path')
parser.add_argument('output_path', type=str, default='./', help='Output Path')

#
parser.add_argument('--making_train_set', type=bool, default=False, help='If you want to make training set : True')
parser.add_argument('--background', type=bool, default=False, help='If you want to make background cube : True')
parser.add_argument('--making_test_data', type=bool, default=False, help='If you want to make SDC3 testdata : True')



parser.add_argument('--epoch', type=int, default=10, help='Epoch')
parser.add_argument('--batch_size', type=int, default=10, help='Epoch')

#
parser.add_argument('--apply_to_SDC3a', type=bool, default=True, help='To apply model to SDC3 : True/False')

if __name == '__main__':
    args=parser.parse_args()
    Mytrainset
    Mytrain
    Mytest
