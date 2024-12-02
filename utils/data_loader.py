from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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
