import torch
import torch.nn as nn
import torch.nn.functional as F


class DNN1(nn.Module):
    def __init__(self):
        super(DNN1, self).__init__()
        # Input layer is implicitly defined by the first linear layer
        self.fc1 = nn.Linear(784, 8)  # 784 inputs (28x28 image), 8 outputs => nodes 784*8=6272
        self.fc2 = nn.Linear(8, 8)  # 8 inputs, 8 outputs => nodes = 64
        self.fc3 = nn.Linear(8, 8)  # 8 inputs, 8 outputs => nodes = 64
        self.fc4 = nn.Linear(8, 10)  # 8 inputs, 10 outputs (digits 0-9) => nodes = 80

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the image
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # No activation, this will be handled by the loss function
        return x


loss_function = nn.CrossEntropyLoss()


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_function(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
