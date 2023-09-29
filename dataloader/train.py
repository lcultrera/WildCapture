import time
import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from dataloader_autoencoder import load_data
from autoencoder import AutoEncoder
from loss_optimizer import create_loss_and_optimizer  # Import the function from loss_optimizer.py

# Load configuration from YAML file
import yaml

with open('config_autoencoder.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

# Read parameters from the configuration file
batch_size = config["batch_size"]
num_epochs = config["num_epochs"]
learning_rate = config["learning_rate"]
img_size = config["img_size"]
data_folder = config["data_folder"]
log_dir = config["log_dir"]

# Create a directory for TensorBoard logs
os.makedirs(log_dir, exist_ok=True)

# Load the data
dataloaders, dataset_sizes = load_data(data_folder, batch_size)

# Create the AutoEncoder model
model = AutoEncoder()

# Move the model to the GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create the loss function and optimizer using the function from loss_optimizer.py
loss_function, optimizer = create_loss_and_optimizer(model, learning_rate)

# Set up TensorBoard
writer = SummaryWriter(log_dir)

# Training loop
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print("-" * 10)

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0

        for inputs, _ in dataloaders[phase]:
            inputs = inputs.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                loss = loss_function(outputs, inputs)

                # Backward pass and optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / dataset_sizes[phase]
        print(f"{phase.capitalize()} Loss: {epoch_loss:.4f}")

        # Log the loss to TensorBoard
        writer.add_scalar(f'{phase.capitalize()} Loss', epoch_loss, epoch)

print("Training finished.")

# Save the trained model
torch.save(model.state_dict(), 'autoencoder_model.pth')

# Close TensorBoard writer
writer.close()
