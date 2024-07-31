# universal-approximation
Using neural networks to predict continuous mathematical functions

This project demonstrates the Universal Approximation Theorem using neural networks to approximate various continuous functions. Implemented using TensorFlow/Keras and PyTorch, the project showcases how neural networks can approximate functions like 
sin(x), exp(x), and composite functions such as exp(x) + x - x**2.

Introduction
The Universal Approximation Theorem states that a feedforward neural network with a single hidden layer containing a finite number of neurons can approximate any continuous function on compact subsets of Real Numbers, under mild assumptions on the activation function.

This project illustrates this theorem by:
1. Generating synthetic datasets.
2. Training neural networks to approximate continuous functions.
3. Visualizing the approximation performance.
4. Performing hyperparameter tuning to optimize model performance.

Project Structure
universal-approximation-theorem/
main.py     # Script for generating synthetic datasets and visualizing results
train.py      # Script for training neural networks using a custom pipeline
dataset.py      # Custom dataset class implementation
models.py      # Custom dataset class implementation
requirements.txt       # Required dependencies
README.md              # Project README

