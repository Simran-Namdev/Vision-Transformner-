# Vision Transformer (ViT) for CIFAR-10 Classification

This repository contains the implementation of a Vision Transformer (ViT) for image classification on the CIFAR-10 dataset using TensorFlow and Keras.

## Table of Contents
- [Dataset](#dataset)
- [Hyperparameter Definition](#hyperparameter-definition)
- [Build ViT Classifier Model](#build-vit-classifier-model)
  - [Data Augmentation](#data-augmentation)
  - [MLP Architecture](#mlp-architecture)
  - [Patches](#patches)
  - [Patch Encoder](#patch-encoder)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Dataset

The CIFAR-10 dataset is used for training and testing the ViT model. It consists of 60,000 32x32 color images in 10 different classes.

## Hyperparameter Definition

The hyperparameters used for training the ViT model are defined in the code, including learning rate, weight decay, batch size, number of epochs, image size, patch size, and various parameters related to the transformer architecture.

## Build ViT Classifier Model

### Data Augmentation

Data augmentation is performed using Keras Sequential API, including normalization, resizing, horizontal flipping, and random zoom.

### MLP Architecture

A multi-layer perceptron (MLP) is defined with customizable hidden units and dropout rates.

### Patches

The input images are divided into patches using a custom `Patches` layer.

### Patch Encoder

A `PatchEncoder` layer is implemented to encode the patches using a combination of dense projection and positional embeddings.

## Training the Model

The ViT model is trained using the CIFAR-10 training set with AdamW optimizer and Sparse Categorical Crossentropy loss. Model checkpoints are saved during training.

## Evaluation

The trained model is evaluated on the CIFAR-10 test set, measuring accuracy and top-5 accuracy.

## Inference

You can use the trained model for inference by providing an image or a set of images. The `img_predict` function is provided to make predictions.

## Results

The training history and evaluation results are presented in the code, showcasing the model's performance over epochs.

## Usage

To use the ViT model for your own tasks, follow the steps outlined in the code. Customize the hyperparameters, architecture, or input data as needed.

## Contributing

Feel free to contribute to the project. If you find any issues or have suggestions, please open an issue or create a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
