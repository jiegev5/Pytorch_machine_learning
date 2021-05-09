import argparse
import torch
import dataloader
from models import main_models
import numpy as np
import sys
import torch
import time

parser=argparse.ArgumentParser()
parser.add_argument('--n_epoches',type=int,default=10)
parser.add_argument('--batch_size',type=int,default=64)

opt=vars(parser.parse_args())

use_cuda=True if torch.cuda.is_available() else False
device=torch.device('cuda:0') if use_cuda else torch.device('cpu')
torch.manual_seed(1)
if use_cuda:
    torch.cuda.manual_seed(1)

#--------------pretrain g and h for step 1---------------------------------
train_dataloader=dataloader.mnist_dataloader(batch_size=opt['batch_size'],train=True)
test_dataloader=dataloader.mnist_dataloader(batch_size=opt['batch_size'],train=False)

# for data,labels in train_dataloader:
#     print(data.shape,labels.shape)

mlp=main_models.MLP(in_feature=784,dropout=0.2,n_hid=256,output_feature=10)
print("mlp: ")
print(mlp)

mlp.to(device)

loss_fn=torch.nn.CrossEntropyLoss()

optimizer=torch.optim.SGD(mlp.parameters(),lr=0.005)


for epoch in range(opt['n_epoches']):
    for data,labels in train_dataloader:

        data=data.to(device)
        labels=labels.to(device)

        optimizer.zero_grad()

        y_pred=mlp(data.view(-1,28*28))

        loss=loss_fn(y_pred,labels)
        loss.backward()

        optimizer.step()

    acc=0
    for data,labels in test_dataloader:
        data=data.to(device)
        labels=labels.to(device)
        y_test_pred=mlp(data.view(-1,28*28))
        acc+=(torch.max(y_test_pred,1)[1]==labels).float().mean().item()

    accuracy=round(acc / float(len(test_dataloader)), 3)

    print("step1----Epoch %d/%d  accuracy: %.3f "%(epoch+1,opt['n_epoches'],accuracy))






















