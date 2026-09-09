# NumPy-Based-Convolutional-Neural-Network-CNN-from-Scratch

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![NumPy](https://img.shields.io/badge/Library-NumPy%20Only-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A lightweight, purely mathematical implementation of a Convolutional Neural Network (CNN) built entirely from scratch using Python and NumPy.

This project demonstrates the fundamental inner workings of deep learning architectures, including forward propagation, backpropagation, and gradient descent, without relying on external deep learning frameworks like TensorFlow or PyTorch.

---

## 🚀 Key Features

* Custom Neural Network Layers: Modular, object-oriented implementations of Conv2D, ReLU, MaxPool2D, and Dense (fully connected) layers.  
* From-Scratch Backpropagation: Every layer includes a custom backward() method to compute and pass gradients through the network.  
* Gradient Stability Measures: Implements a stabilized Softmax Cross-Entropy loss function (preventing overflow/underflow) and a gradient distribution fix for tied maximum values in the MaxPool2D layer.  
* Automated Data Loading: Includes a built-in directory scanner that automatically resizes images, normalizes pixel values to a [0,1] scale, dynamically handles RGB or Grayscale channel mapping, and labels data based on folder structures.  
* Checkpointing: Utility functions to save trained model weights to a compressed .npz file and reload them for future inference or continued training.

---

## 🛠️ Installation & Prerequisites

This implementation is designed to have minimal dependencies. Ensure you have the following installed in your Python environment:  
* numpy (For matrix operations and network math)  
* Pillow (For image loading and resizing via PIL.Image)
```
pip install numpy pillow
```

---

## ⚙️ How to Run

1. Prepare the Dataset: Organize your image data (.png, .jpg, .jpeg) into the dataset/train and dataset/test folder. Ensure your dataset folder is placed in the root directory.  
2. Execute the Script: Run the main Python file to begin the training pipeline.  
```
python numpyCNN.py
```
3. Training Pipeline:  
* The script will verify and load data from both the train and test directories.  
* It will initialize the model layers using random distributions (np.random.randn) for weights/filters and zeros for biases.  
* It shuffles the training dataset dynamically at the start of each epoch.  
* During training, it prints the average loss, training accuracy, and testing accuracy per epoch.  
4. Save/Load Weights: Upon completion of the training loop, the network parameters are automatically saved to cnn_weights.npz. To resume training from a saved state, uncomment the load_weights("cnn_weights", conv, dense) line in numpyCNN.py.  

---

## 🎛️ Hyperparameter Customization

The network architecture and training configurations are defined at the top of numpyCNN.py and can be easily adjusted.  

Dataset & Input Dimensions  
INPUT_HEIGHT = 32      # Target image height for resizing during load  
INPUT_WIDTH = 32       # Target image width for resizing during load  
IN_CHANNELS = 3        # Color channels (3 for RGB, 1 for Grayscale)  
NUM_CLASSES = 4        # Total distinct categories/folders to classify  

Model Architecture Settings  
CONV_OUT_CHANNELS = 4  # Number of feature detectors (filters) in Conv2D  
CONV_KERNEL = 3        # Sliding window size (3x3) for Conv2D  
POOL_SIZE = 2          # Downsampling window size (2x2) for MaxPool  

Training Parameters  
LEARNING_RATE = 0.01   # Step size for weight updates during backprop  
EPOCHS = 10            # Total complete passes over the entire dataset  
BATCH_SIZE = 32        # Number of images processed simultaneously
