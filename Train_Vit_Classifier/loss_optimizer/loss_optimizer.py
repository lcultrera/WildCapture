import torch
import torch.nn as nn
import torch.optim as optim

def create_loss_optimizer(model, learning_rate):
    """
    Create a loss function and optimizer for training.

    Args:
        model (nn.Module): The neural network model.
        learning_rate (float): The learning rate for the optimizer.

    Returns:
        criterion (nn.Module): The loss function.
        optimizer (optim.Optimizer): The optimizer.
    """
    # Define the loss function (e.g., CrossEntropyLoss for classification)
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer (e.g., Adam optimizer with the specified learning rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    return criterion, optimizer

