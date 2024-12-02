from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np

def create_dataloaders(data_dir, batch_size=32):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    train_dataset = datasets.ImageFolder(root=f'{data_dir}/train', transform=data_transforms['train'])
    val_dataset = datasets.ImageFolder(root=f'{data_dir}/val', transform=data_transforms['val'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    dataloaders = {'train': train_loader, 'val': val_loader}
    return dataloaders

def create_test_loader(test_split=0.2, batch_size=32):
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load CIFAR10 test set
    cifar_test = datasets.CIFAR10(root="./", train=False, download=True, transform=test_transform)

    # Simulate CIFAKE10 labels: Classes 0-4 are "authentic" (0), 5-9 are "AI-generated" (1)
    cifar_test.targets = [0 if label < 5 else 1 for label in cifar_test.targets]

    # Subset the test set into 20% for testing
    labels = np.array(cifar_test.targets)
    indices = np.arange(len(labels))
    _, test_indices = train_test_split(indices, test_size=test_split, stratify=labels, random_state=42)

    test_subset = Subset(cifar_test, test_indices)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=4)

    return test_loader


