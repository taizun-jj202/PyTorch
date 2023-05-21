import torch
import torch.nn as nn
import torch.optim as optim     #All optimization algorithms like SGD, Adam
import torch.nn.functional as F #Contains activation functions, relu, tanh
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class CNN(nn.Module):

    def __init__(self,input_size = 1, num_classes = 10):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8,kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(16*7*7, num_classes) #Fully conncected layer

    def forward(self,x) :
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x

model = CNN()
x = torch.rand(64,1,28,28)
print(model(x).shape)

#Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameters:
in_channel = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5

#Loading and dividing the data
train_dataset = datasets.MNIST(root='dataset/',train = True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle= True)

test_dataset = datasets.MNIST(root='dataset/',train = False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle= True)


#Initialize network
model = CNN().to(device)

#Loss and optimizer function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

#Training the netowrk

for epoch in range(num_epochs):
    for batch_idx,(data,index) in enumerate(train_loader):
        data = data.to(device = device)
        index = index.to(device = device)

        #Forward pass
        scores = model(data)
        loss = criterion(scores, index)

        #Backwards
        optimizer.zero_grad()
        loss.backward()
        #Loss function
        optimizer.step()

#Checking the accuracy :
def accuracy(loader,model):
    if loader.dataset.train :
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on testing data")

    num_correct = 0
    num_passes = 0

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device = device)
            y = y.to(device = device)

            scores = model(x)
            _ , pred = scores.max(1)
            num_correct += (pred == y).sum()
            num_passes += pred.size(0)
        print(f"Got {num_correct}/{num_passes} with accuracy {float(num_correct)/float(num_passes)*100}")

    model.train()


accuracy(train_loader, model)
accuracy(test_loader, model)