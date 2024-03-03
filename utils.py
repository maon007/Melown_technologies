import torch

def calculate_mean_std(data_dir):
    # Define calculate_mean_std function here

def get_dataloaders(train_set, val_set, batch_size=32):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
