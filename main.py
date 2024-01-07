from NN import DNN1, train, test
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
# todo make the implementation simplier
# todo move dataloader out
# todo provide visual implementation in jupyter notebook
# todo add final requirements file of env setup
# todo move state dict into a folder save/
epochs = 5 # set how many iteration for training

# Define the neural network class
model = DNN1()


# Define the optimizer and loss function
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Custom transform using Lambda
min_max_scaler = transforms.Lambda(lambda x: (x - torch.min(x)) / (torch.max(x) - torch.min(x)))

# Add to your transform pipeline
transform = transforms.Compose([
    transforms.ToTensor(),
    min_max_scaler
])

# # Load and preprocess the MNIST dataset
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])


train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Training the model


# Training settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



# Train for 50 epochs as mentioned in the paper
# epoch in range(1, 51):
for epoch in range(1, epochs):
    train(model, device, train_loader, optimizer, epoch)

# Evaluate the model
test(model, device, test_loader)


# save model
torch.save(model.state_dict(),'torch_model_state_dict')
# import numpy as np
# list(map(np.shape, model.parameters()))