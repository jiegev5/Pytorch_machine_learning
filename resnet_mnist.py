import argparse
import torch
import dataloader
from models import main_models
import numpy as np
import sys
import torch
import time
from torchsummary import summary
import matplotlib.pyplot as plt
import seaborn as sns

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
cls_num=10
encoder = main_models.resnet_encoder("resnet18",pretrained=False)
model=main_models.ResNet_MNIST(encoder,num_classes=cls_num)

# visual model size at each layer
summary(encoder,(1,28,28),batch_size=-1,device='cpu')

model.to(device)
loss_fn=torch.nn.CrossEntropyLoss()

optimizer=torch.optim.Adam(model.parameters())

trainLoss, testLoss = [],[]
train_loss_min = np.inf

for epoch in range(opt['n_epoches']):

    tloss = 0.0
    valloss = 0.0

    model.train()

    for data,labels in train_dataloader:

        data=data.to(device)
        labels=labels.to(device)

        optimizer.zero_grad()

        y_pred=model(data)

        loss=loss_fn(y_pred,labels)
        loss.backward()

        optimizer.step()

    acc=0
    model.eval()
    for data,labels in test_dataloader:
        data=data.to(device)
        labels=labels.to(device)
        y_test_pred=model(data)
        acc+=(torch.max(y_test_pred,1)[1]==labels).float().mean().item()

    accuracy=round(acc / float(len(test_dataloader)), 3)

    print("step1----Epoch %d/%d  accuracy: %.3f "%(epoch+1,opt['n_epoches'],accuracy))

    # save best model
    if tloss <= train_loss_min:
        print('Train loss decreased ({:.3f} --> {:.3f}).  Saving model ...'.format(
        train_loss_min,
        tloss))
        torch.save(model.state_dict(), 'saved_models/model_spec.pt')
        train_loss_min = tloss    

# save model
# torch.save(model.state_dict(), 'saved_models/model_spec.pt')
model.load_state_dict(torch.load('saved_models/model_spec.pt'));

# Plot the resulting loss over time
# plt.plot(trainLoss, label='Training Loss')
# plt.plot(testLoss, label='Testing Loss')
# plt.legend()
# plt.show()

# generate confusion matrix

class_correct = [0]*cls_num
class_total   = [0]*cls_num
conf_matrix = np.zeros((cls_num,cls_num))
model.eval()

for data,labels in test_dataloader:
    data=data.to(device)
    labels=labels.to(device)
    y_test_pred=model(data)
    # acc+=(torch.max(y_test_pred,1)[1]==labels).float().mean().item()

    _, pred = torch.max(y_test_pred, 1)  
    # compare predictions to true label
    correct_tensor = pred.eq(labels.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if device == "cpu" else np.squeeze(correct_tensor.cpu().numpy())

    # Update confusion matrix
    for i in range(labels.size(0)):
        label = labels.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1
        
        # Update confusion matrix
        conf_matrix[label][pred.data[i]] += 1

# print accuracy for each class
for i in range(cls_num):
    if class_total[i] > 0:
        print('Test Accuracy of %3s: %2d%% (%2d/%2d)' % (
            i, 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %3s: N/A (no training examples)' % (i))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))

plt.subplots(figsize=(10,10))
ax = sns.heatmap(conf_matrix, annot=True, square=True, vmax=20)
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label')
ax.set_aspect("equal")
plt.tight_layout()
plt.show()


sys.exit()


















