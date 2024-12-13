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
        nn.BatchNorm1d(256),  # Batch Normalization
        nn.Dropout(0.3),
        nn.Linear(256, 128),  # New layer added
        nn.ReLU(),
        nn.BatchNorm1d(128),  # Batch Normalization for the new layer
        nn.Dropout(0.3),
        nn.Linear(128, num_classes)  # Output layer
    )
    return densenet
