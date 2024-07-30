# Speech Command Recognition with Deep Learning

This project leverages deep learning to accurately recognize and classify speech commands from audio inputs. Utilizing the PyTorch library and the Google Speech Commands dataset, it features a convolutional neural network (CNN) designed to process short audio clips and categorize them into a set of predefined commands.

## Features

- Utilizes PyTorch for constructing and training the deep learning model.
- Incorporates the Google Speech Commands dataset for comprehensive training and evaluation.
- Includes data preprocessing, model training, validation, and testing phases.
- Produces confusion matrices, and accuracy and loss plots for training and validation phases.
- Designed to be compatible with both CPU and GPU for computational efficiency.

## Dependencies

The following Python packages are required:

run `pip install -r requirements.txt` to install the required packages.

## Usage

Before running the script, ensure all dependencies are installed. The script can automatically download the Google Speech Commands dataset if it's not already present locally. To run the script:

```bash
python main.py
```

## Model Architecture

The `DeepGSC` model is a CNN that transforms audio waveforms into spectrograms for speech command classification. The architecture comprises multiple convolutional, batch normalization, and ReLU activation layers, followed by max pooling and fully connected layers, culminating in a softmax output layer for class prediction.

## Dataset

The Google Speech Commands dataset is employed, consisting of one-second long audio clips of spoken words. These clips are preprocessed into a suitable format for training.

## Results

Upon completion, the model's performance metrics on the test set will be displayed, including accuracy and a confusion matrix. The best model parameters are saved for future inference.


