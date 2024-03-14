# 3D u-NET based Deep Learning for Point Source Removal

## Project Overview
This project introduces deep learning models for point source removal based on the SKA Science Data Challenge 3a. It utilizes a 3D u-NET architecture to effectively identify and remove point sources from astronomical data, potentially playing a significant role in astronomical research.

## Prerequisites
Before starting, ensure you have Python 3.7, TensorFlow 2.4, Astropy 4.0.1, Healpy 1.14.0, and Numpy 1.16.4 installed. The project is designed to leverage GPU acceleration for deep learning model training, so a GPU-enabled setup is recommended.

## Installation
Clone the project to your local system and install the required dependencies as follows:

```bash
git clone https://your-project-repository-url.git
cd your-project-directory
pip install -r requirements.txt
```
## Usage
Run the main script of the project like this:

```bash
python main.py <input_path> <output_path> [OPTIONS]
```
## Required Arguments
input_path: Path to the input data.

output_path: Path where the output will be saved.
## Optional Arguments

--making_train_set: Flag to trigger the creation of the training set.

-n, --number_of_trainset: Specify the number of training sets (default: 2560).

-a, --augmentation: Enable rotational augmentation (augments the number of training sets by 4).

-b, --background: Flag to create a background cube.

--making_test_data: Flag to generate SDC3 test data.

-f, --frequency_channels: Number of frequency channels (default: 264, with 1 channel = 0.1MHz).

-e, --epochs: Number of training epochs (default: 10).

-v, --validation_split: Fraction of the data to be used as validation set (default: 0.2).

--batch_size: Batch size for training (default: 4).

--apply_to_SDC3a: Flag to apply the model to SDC3 test data.

Replace <input_path> and <output_path> with your desired file paths and use the optional arguments as needed for your specific requirements.

## License
This project is distributed under the MIT License.

## Authors and Contributors
Namuk Lee, dlskadnr1209@unist.ac.kr
