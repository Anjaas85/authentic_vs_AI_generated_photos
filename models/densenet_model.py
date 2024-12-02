import torch.nn as nn
from torchvision import models

def create_densenet(num_classes=2):
    # Load pretrained DenseNet121
    densenet = models.densenet121(pretrained=True)
    
    # Replace the classifier
    num_features = densenet.classifier.in_features
    densenet.classifier = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
        nn.Softmax(dim=1)
    )
    return densenet
