import argparse
import torch
import dataloader
from models import main_models
import numpy as np
import sys
import torch
import time
from torchsummary import summary
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
#     print(f'mnist dataset: mean: {data.mean()} max: {data.max()} min: {data.min()} std: {data.std()}')

#     sys.exit()

classifier=main_models.Classifier()
print("classifier: ")
print(classifier)
encoder=main_models.Encoder()
print("Encoder: ")
print(encoder)

# visual model size at each layer
summary(encoder,(1,28,28),batch_size=-1,device='cpu')

classifier.to(device)
encoder.to(device)
loss_fn=torch.nn.CrossEntropyLoss()

optimizer=torch.optim.Adam(list(encoder.parameters())+list(classifier.parameters()))


for epoch in range(opt['n_epoches']):
    for data,labels in train_dataloader:

        data=data.to(device)
        labels=labels.to(device)

        optimizer.zero_grad()

        y_pred=classifier(encoder(data))

        loss=loss_fn(y_pred,labels)
        loss.backward()

        optimizer.step()

    acc=0
    for data,labels in test_dataloader:
        data=data.to(device)
        labels=labels.to(device)
        y_test_pred=classifier(encoder(data))
        acc+=(torch.max(y_test_pred,1)[1]==labels).float().mean().item()

    accuracy=round(acc / float(len(test_dataloader)), 3)

    print("step1----Epoch %d/%d  accuracy: %.3f "%(epoch+1,opt['n_epoches'],accuracy))

sys.exit()


















