import torch
import numpy as np

class EarlyStopping:
    """
    Early stopping utility:
    - Stops training when validation loss doesn't improve after a given patience.
    - Saves the best model checkpoint.
    """
    def __init__(self, patience=10, checkpoint_path='best_model.pth'):
        self.patience = patience
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
        self.checkpoint_path = checkpoint_path

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            # Save the best model checkpoint
            torch.save(model.state_dict(), self.checkpoint_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
