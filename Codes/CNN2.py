import torch
import torch.nn as nn
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 =nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

    
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1))
        
        self.fc1 = nn.Linear(4 * 4 * 128, 625, bias=True) # fully connected, 
        nn.init.kaiming_uniform_(self.fc1.weight)
        self.layer4 = nn.Sequential(
            self.fc1,
            nn.ReLU())
        self.fc2 = nn.Linear(625, 9, bias=True)
        nn.init.kaiming_uniform_(self.fc2.weight)      
        
    def printout(self, x):
        out = self.layer1(x)
        print(out)
        out = self.layer2(out)
        print(out)
        out = self.layer3(out)
        print(out)
        out = out.view(out.size(0), -1)  
        out = self.layer4(out)
        print(out)
        out = self.fc2(out)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.layer4(out)
        out = self.fc2(out)
        return out    