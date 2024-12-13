import torch.nn as nn
from torchvision import models

def create_densenet(num_classes=2):
    densenet = models.densenet121(pretrained=True)
    num_features = densenet.classifier.in_features
    densenet.classifier = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),  # Batch Normalization
        nn.Dropout(0.3), 
        nn.Linear(256, num_classes)
    )
    return densenet

"""
There is no Softmax at the end of the DenseNet model because:

CrossEntropyLoss Expectation:
PyTorch's nn.CrossEntropyLoss combines LogSoftmax and NLLLoss in one function. 
It expects raw logits as input, not probabilities. Adding Softmax would cause 
incorrect scaling and degrade performance.

Numerical Stability:
Applying Softmax before the loss can introduce numerical instability due to 
very small or large values in logits, risking overflow or underflow issues.

"""