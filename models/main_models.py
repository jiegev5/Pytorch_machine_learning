import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BasicModule import BasicModule
from torch_same_pad import get_pad, pad # pip install git+https://github.com/CyberZHG/torch-same-pad.git
from torchsummary import summary

class DCD(BasicModule):
    def __init__(self,h_features=64,input_features=128):
        super(DCD,self).__init__()

        self.fc1=nn.Linear(input_features,h_features)
        self.fc2=nn.Linear(h_features,h_features)
        self.fc3=nn.Linear(h_features,4)

    def forward(self,inputs):
        out=F.relu(self.fc1(inputs))
        out=self.fc2(out)
        return F.softmax(self.fc3(out),dim=1)

class Classifier(BasicModule):
    def __init__(self,input_features=64):
        super(Classifier,self).__init__()
        self.fc=nn.Linear(input_features,10)

    def forward(self,input):
        return F.softmax(self.fc(input),dim=1)

class Encoder(BasicModule):
    def __init__(self):
        super(Encoder,self).__init__()

        self.conv1=nn.Conv2d(1,6,5)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(256,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,64)

    def forward(self,input):
        out=F.relu(self.conv1(input))
        out=F.max_pool2d(out,2)
        out=F.relu(self.conv2(out))
        out=F.max_pool2d(out,2)
        out=out.view(out.size(0),-1)

        out=F.relu(self.fc1(out))
        out=F.relu(self.fc2(out))
        out=self.fc3(out)

        return out

""" class model of target network for testing """

class CNN_SPEC(BasicModule):
    def __init__(self):
        super(CNN_SPEC, self).__init__()

        self.num_classes = 4
        self.in_channels = 1
        self.dropout = 0.4
        self.input_size = [32,10]
        self.fc_in = 512 # 32*int((self.input_size[0]/4-3)*(self.input_size[1]/4-3))

        self.conv1 = nn.Sequential(
                nn.Conv2d(self.in_channels, 16, 4, padding=(1,1), stride=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((2, 2),stride=2)
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(16, 32, 2, padding=(1,1), stride=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((2, 2),stride=2)
                )
        self.fc = nn.Sequential(
                nn.Linear(self.fc_in, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout),
                nn.Linear(1024, self.num_classes)
                )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1,self.fc_in)
        x = self.fc(x)
        return x

class DCD_GTSRB(BasicModule):
    def __init__(self,h_features=64,input_features=128):
        super(DCD,self).__init__()

        self.fc1=nn.Linear(input_features,h_features)
        self.fc2=nn.Linear(h_features,h_features)
        self.fc3=nn.Linear(h_features,4)

    def forward(self,inputs):
        out=F.relu(self.fc1(inputs))
        out=self.fc2(out)
        return F.softmax(self.fc3(out),dim=1)

class Classifier_GTSRB(BasicModule):
    def __init__(self,input_features=64):
        super(Classifier_GTSRB,self).__init__()
        self.fc=nn.Linear(input_features,43)

    def forward(self,input):
        # return F.softmax(self.fc(input),dim=1)
        return self.fc(input)

class Encoder_GTSRB(BasicModule):
    def __init__(self):
        super(Encoder_GTSRB,self).__init__()

        self.conv1=nn.Conv2d(3,32,5)
        self.conv2=nn.Conv2d(32,32,5)
        # self.fc1=nn.Linear(400,120)
        # self.fc2=nn.Linear(120,84)
        # self.fc3=nn.Linear(84,64)
        self.fc=nn.Linear(800,64)
        # self.fc2=nn.Linear(128,64)

    def forward(self,input):
        out=F.relu(self.conv1(input))
        out=F.max_pool2d(out,2)
        out=F.relu(self.conv2(out))
        out=F.max_pool2d(out,2)
        out=out.view(out.size(0),-1)
        # print(out.shape)
        # out=F.relu(self.fc1(out))
        # out=F.relu(self.fc2(out))
        # out=self.fc3(out)
        # out=F.relu(self.fc1(out))
        out=self.fc(out)

        return out

class MLP(BasicModule):
    def __init__(self, in_feature=100, dropout=0.2,n_hid=256,output_feature=10):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_feature, n_hid),
            nn.ReLU(),
            nn.Linear(n_hid, n_hid*2),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(n_hid*2, n_hid*4),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(n_hid*4, output_feature),
        )

    def forward(self, input_tensor):
        # Concatenate label embedding and image to produce input
        # validity = self.model(input_tensor)
        return self.model(input_tensor)




