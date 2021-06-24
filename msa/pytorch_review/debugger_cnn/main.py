import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

train_set = torchvision.datasets.FashionMNIST(
    root="./data/FashionMNIST",
    train=True,
    download=True,
    transform=transforms.Compose([ # convert image to 
        transforms.ToTensor()
    ]))

image, label = train_set[0]

print(image.shape)

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional layers
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1) # in_channel = 1 = grayscale, hyperparam, hyperparam
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5, stride=1) # we in crease the output channel when have extra conv layers
                
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120, bias=True) # we also shrink the number of features to number of class that we have
        self.fc2 = nn.Linear(in_features = 120, out_features=60, bias=True)
        self.out = nn.Linear(in_features = 60, out_features=10, bias=True) 
        
    def forward(self, t):
        # input layer
        t = t
        
        # convolution 1, not 
        t = self.conv1(t)
        t = F.relu(t) # operation do not use weight, unlike layers
        t = F.max_pool2d(t, kernel_size=2, stride=2) # operation do not use weight, unlike layers
        
        # convolution 2: => relu => maxpool
        t = self.conv2(t)
        # WHY do we need these 2 layers?
        t = F.relu(t) 
        t = F.max_pool2d(t, kernel_size=2, stride=2) # how to determine these values?
        
        # Transition from Conv to Linear will require flatten
        t = t.reshape(-1, 12*4*4) # 4x4 = shape of reduce image (originally 28x28)
        
        # linear 1:
        t = self.fc1(t)
        t = F.relu(t)
        
        # linear 2:
        t = self.fc2(t)
        t = F.relu(t)
        
        # output:
        t = self.out(t)
        
        return t

def main():
        
    network = Network()

    train_loader = torch.utils.data.DataLoader(train_set, batch_size = 100)
    optimizer = optim.Adam(network.parameters(), lr=0.01)

    for epoch in range(5):

        total_loss = 0
        total_correct = 0

        for i, batch in enumerate(train_loader):
        #     print(f"Batch {i}")
            images, labels = batch

            preds = network(images)
            loss = F.cross_entropy(preds, labels) # calculate loss

            # Each weight has the corresponsing Gradient
            # before we calculate a new gradient for the same weight via each batch we have to zero out the gradient.
            # we want to use the new calculated gradient to update the weight. 

            optimizer.zero_grad() 
            loss.backward() # calculate gradient/ backprop. Note, this does not affect the loss but just the learning hyperparam
            optimizer.step() # Update the weight

            total_loss += loss.item()
            total_correct += get_num_correct(preds, labels)

        print(f"epoch: {epoch}, total_correct: {total_correct}, loss: {total_loss}")

    print(f"The accuracy rate is {total_correct / len(train_set)}") # total correct of the latest trained model

if __name__ == "__main__":
    main()