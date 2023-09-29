import time
import os
import torch
import torch.optim as optim
from dataloader.dataloader import get_validationSet
from model.AutoEncoder import AutoEncoder
from loss_optimizer.loss_optimizer import create_loss_and_optimizer  # Import the function from loss_optimizer.py
from tqdm import tqdm
import torch.nn.functional as F
import pickle
# Load configuration from YAML file
import yaml


with open('config/Test_config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

# Read parameters from the configuration file
batch_size = config["batch_size"]
model_path = config["model_path"]
dest_file = config["dest_file"]


ReconstructionErrorList = []


# Load the data
print("preparing the dataset")
validation_generator = get_validationSet()
print("dataset loaded")
# Create the AutoEncoder model
model = AutoEncoder()
print("model created")
# Move the model to the GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

model.load_state_dict(torch.load(model_path)) #MEGLIO

loss, optimizer = create_loss_and_optimizer(model, learning_rate)

# Set up TensorBoard

# Training loop
n_batches = len(train_generator)

Val_running_loss = 0


for i, data in tqdm(enumerate(validation_generator)):
  inputs, labels = data
  img_tensor = inputs
  img_tensor = img_tensor.cuda()
  inp = img_tensor.float()
  outputs = model(inp)

  Val_loss_size =  loss(outputs,inp)
  Val_running_loss = Val_loss_size.data
  ReconstructionErrorList.append(Val_running_loss.detach().cpu().numpy())
  
with open(dest_file, "wb") as fp:
    pickle.dump(ReconstructionErrorList, fp)
print("Training finished, took {:.8f}s".format(time.time() - training_start_time))
