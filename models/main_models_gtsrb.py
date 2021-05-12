import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BasicModule import BasicModule
class DCD(BasicModule):
    def __init__(self,h_features=800,input_features=1600):
        super(DCD,self).__init__()

        self.fc1=nn.Linear(input_features,h_features)
        self.fc2=nn.Linear(h_features,h_features)
        self.fc3=nn.Linear(h_features,4)

    def forward(self,inputs):
        out=F.relu(self.fc1(inputs))
        out=self.fc2(out)
        return F.softmax(self.fc3(out),dim=1)

class Classifier(BasicModule):
    def __init__(self,input_features=800):
        super(Classifier,self).__init__()
        self.fc=nn.Linear(input_features,43)

    def forward(self,input):
        return F.softmax(self.fc(input),dim=1)

class Encoder(BasicModule):
    def __init__(self):
        super(Encoder,self).__init__()
        
        self.input_size=32

        self.conv1 = nn.Sequential(
                nn.Conv2d(3, 32, 5, stride=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(32, 32, 5, stride=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                )
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 32*int((self.input_size/4-3)*(self.input_size/4-3)))

        # out=F.relu(self.fc1(out))
        # out=F.relu(self.fc2(out))
        # out=self.fc3(out)

        return x





