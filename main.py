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
parser.add_argument('--making_background', type=bool, default=False, help='To make background cube : True/False')
parser.add_argument('--making_test_data', type=bool, default=False, help='To make test data : True/False')
parser.add_argument('--epoch', type=int, default=10, help='Epoch')

#


if __name == '__main__':
    args=parser.parse_args()
    
