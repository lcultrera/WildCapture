import time
import os
import torch
import torch.optim as optim
from dataloader.dataloader import get_dataset
from model.AutoEncoder import AutoEncoder
from loss_optimizer.loss_optimizer import create_loss_and_optimizer  # Import the function from loss_optimizer.py
from tqdm import tqdm
import torch.nn.functional as F

# Load configuration from YAML file
import yaml

with open('config/config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

# Read parameters from the configuration file
batch_size = config["batch_size"]
num_epochs = config["num_epochs"]
learning_rate = config["learning_rate"]

# Create a directory for TensorBoard logs

# Load the data
print("preparing the dataset")
train_generator, validation_generator = get_dataset()
print("dataset loaded")
# Create the AutoEncoder model
model = AutoEncoder()
print("model created")
# Move the model to the GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create the loss function and optimizer using the function from loss_optimizer.py
loss, optimizer = create_loss_and_optimizer(model, learning_rate)

# Set up TensorBoard

# Training loop
n_batches = len(train_generator)
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print("-" * 10)
    start_time = time.time()
    running_loss = 0.0
    total_train_loss = 0
    Val_total_train_loss = 0
    Val_running_loss = 0
    print_every = n_batches // 10

    for i, data in tqdm(enumerate(train_generator)):
        inputs, labels = data
        optimizer.zero_grad()
        img_tensor = inputs
        img_tensor = img_tensor.cuda()
        inp = img_tensor.float()
        outputs = model(inp)

        loss_size =  loss(outputs,inp)
        loss_size.backward()
        optimizer.step()
        # Print statistics
        running_loss += loss_size.data
        total_train_loss += loss_size.data
        # Print every 10th batch of an epoch
        if (i + 1) % (print_every + 1) == 0:
            print("Epoch {}, {:d}% \t train_loss: {:.8f} took: {:.8f}s".format(
                (epoch) + 1, int(100 * (i + 1) / n_batches), running_loss / print_every, time.time() - start_time))
            # Reset running loss and time
            running_loss = 0.0
            start_time = time.time()

    if(epoch %1 == 0):
        torch.save(model.state_dict(), "model-weights-{}.pt".format(epoch))

    for i, data in tqdm(enumerate(validation_generator)):
            # Get inputs
        inputs, labels = data
            # Set the parameter gradients to zero

            # Forward pass, backward pass, optimize
        img_tensor = inputs
        img_tensor = img_tensor.cuda()
        inp = img_tensor.float()
        outputs = model(inp)

            # Forward pass
        Val_loss_size =  loss(outputs,inp)
            # Print statistics
        Val_running_loss += Val_loss_size.data
        Val_total_train_loss += Val_loss_size.data

    print("MSE train = {:.8f}".format(total_train_loss / len(train_generator)))

    print("MSE Val = {:.8f}".format(Val_total_train_loss / len(validation_generator)))

print("Training finished.")

