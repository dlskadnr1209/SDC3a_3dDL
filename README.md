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
python main.py <input_path> <output_path> --epochs 10 --batch_size 4
```

### Replace <input_path> and <output_path> with your desired file paths and specify additional options as required.

## License
This project is distributed under the MIT License.

## Authors and Contributors
Namuk Lee, dlskadnr1209@unist.ac.kr
