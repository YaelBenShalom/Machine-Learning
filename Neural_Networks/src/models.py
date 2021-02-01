import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import ReLU, Linear


# Please read the free response questions before starting to code.
#
# Note: Avoid using nn.Sequential here, as it prevents the test code from
# correctly checking your model architecture and will cause your code to
# fail the tests.

class Digit_Classifier(nn.Module):
    """
    This is the class that creates a neural network for classifying handwritten digits
    from the MNIST dataset.

    Network architecture:
    - Input layer
    - First hidden layer: fully connected layer of size 128 nodes
    - Second hidden layer: fully connected layer of size 64 nodes
    - Output layer: a linear layer with one node per class (in this case 10)

    Activation function: ReLU for both hidden layers

    """

    def __init__(self):
        super(Digit_Classifier, self).__init__()
        size_layer1 = 128
        size_layer2 = 64
        self.hidden1 = nn.Linear(28 * 28, size_layer1)
        self.hidden2 = nn.Linear(size_layer1, size_layer2)
        self.out = nn.Linear(size_layer2, 10)

    def forward(self, inputs):
        hidden1_out = F.relu(self.hidden1(inputs))
        hidden2_out = F.relu(self.hidden2(hidden1_out))
        output = self.out(hidden2_out)

        return output


class Dog_Classifier_FC(nn.Module):
    """
    This is the class that creates a fully connected neural network for classifying dog breeds
    from the DogSet dataset.

    Network architecture:
    - Input layer
    - First hidden layer: fully connected layer of size 128 nodes
    - Second hidden layer: fully connected layer of size 64 nodes
    - Output layer: a linear layer with one node per class (in this case 10)

    Activation function: ReLU for both hidden layers

    """

    def __init__(self) -> object:
        super(Dog_Classifier_FC, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        size_layer1 = 128
        size_layer2 = 64
        self.hidden1 = nn.Linear(12288, size_layer1)
        self.hidden2 = nn.Linear(size_layer1, size_layer2)
        self.out = nn.Linear(size_layer2, 10)

    def forward(self, inputs):
        hidden1_out = F.relu(self.hidden1(inputs))
        hidden2_out = F.relu(self.hidden2(hidden1_out))
        output = self.out(hidden2_out)

        return output

