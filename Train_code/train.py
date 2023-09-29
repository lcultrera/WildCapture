import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.optim import lr_scheduler
from dataLoader import load_data  # Import your DataLoader function
from VitModel import VitModel  # Import your VitModel class
from loss_optimizer import create_loss_optimizer  # Import your loss and optimizer function
import yaml
import tqdm

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs, device):
    """
    Train the model.

    Args:
        model (nn.Module): The neural network model.
        dataloaders (dict): Dictionary of DataLoader objects for 'train' and 'val' sets.
        dataset_sizes (dict): Dictionary containing the sizes of 'train' and 'val' sets.
        criterion (nn.Module): The loss function.
        optimizer (optim.Optimizer): The optimizer.
        scheduler (lr_scheduler._LRScheduler): Learning rate scheduler.
        num_epochs (int): Number of training epochs.
        device (str): Device to run training on ('cuda' or 'cpu').

    Returns:
        best_model_wts (dict): The weights of the best model.
    """
    best_model_wts = None
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm.tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    return best_model_wts

if __name__ == "__main__":
    
    # Define the path to the YAML config file
    config_path = 'config.yaml'

    # Load the config file
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Extract parameters from the config
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    num_epochs = config["num_epochs"]
    learning_rate = config["learning_rate"]
    img_size = config["img_size"]
    data_dir = config["data_folder"]
    weighted_sampling = config["weighted_sampling"]
    train_split = config["train_split"]

    # Set device (cuda or cpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load data using your DataLoader function
    dataloaders, dataset_sizes, class_names = load_data(data_dir, batch_size, num_workers, img_size, weighted_sampling, train_split)

    # Create an instance of your VitModel class
    model = VitModel(num_classes=len(class_names))
    model = model.to(device)

    # Create the loss function and optimizer using your loss_optimizer function
    criterion, optimizer = create_loss_optimizer(model, learning_rate)

    # Create a learning rate scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Train the model
    best_model_wts = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs, device)

    # Save the best model weights
    torch.save(best_model_wts, 'best_model_weights.pth')

