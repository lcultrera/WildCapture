import torch.optim as optim
import torch.nn as nn

def create_loss_and_optimizer(net, learning_rate=0.0001):
    """
    Create and return the loss function and optimizer for the autoencoder model.

    Args:
        net (nn.Module): The autoencoder model.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        tuple: A tuple containing the loss function and optimizer.
    """
    # Loss function (MSE loss)
    loss_function = nn.MSELoss()

    # Optimizer (Adam optimizer)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    return loss_function, optimizer


