import torch
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data.dataset import Dataset
import pickle

def load_mnist(args):
    # torch.cuda.manual_seed(1)
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': False}
    path = 'data_m/'
    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])),
                # batch_size=32, shuffle=False, **kwargs)
                batch_size=32, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size=1, shuffle=True, **kwargs)
            # batch_size=32, shuffle=True, **kwargs)

    return train_loader, test_loader

def load_batch(fpath):
    images = []
    labels = []
    num_classes = 43
    with open(fpath, 'rb') as rfile:
        train_dataset =  pickle.load(rfile)
    for image in train_dataset['features']:
        # print(image.min(),image.max())
        images.append((image/255)-.5)
    for label in train_dataset['labels']:
        # labels.append(np.eye(num_classes)[label])
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels

class readGTSRB:
    def __init__(self):
        self.train_data = []
        self.train_labels = []
        self.test_data = []
        self.test_labels = []
        self.validation_data = []
        self.validation_labels = []

        # load train data

        img, lab = load_batch('./data/traffic-signs-data/train.p')
        self.train_data.extend(img)
        self.train_labels.extend(lab)

        self.train_data = np.array(self.train_data,dtype=np.float32)
        self.train_labels = np.array(self.train_labels)    

        # load test data

        img, lab = load_batch('./data/traffic-signs-data/test.p')
        self.test_data.extend(img)
        self.test_labels.extend(lab)

        self.test_data = np.array(self.test_data,dtype=np.float32)
        self.test_labels = np.array(self.test_labels)   

        # load validation data

        img, lab = load_batch('./data/traffic-signs-data/valid.p')
        self.validation_data.extend(img)
        self.validation_labels.extend(lab)

        self.validation_data = np.array(self.validation_data,dtype=np.float32)
        self.validation_labels = np.array(self.validation_labels)  

class GTSRB(Dataset):
    def __init__(self,mode):
        self.data = torch.from_numpy(getattr(readGTSRB(),'{}_data'.format(mode))).float().permute(0,3,1,2)
        # print(self.data.shape)
        self.target = torch.from_numpy(getattr(readGTSRB(),'{}_labels'.format(mode))).long()
        # print(self.target[0])
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]        
        return x, y
    
    def __len__(self):
        return len(self.data)

def load_gtsrb():
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': False}
    train_loader = torch.utils.data.DataLoader(GTSRB('train'),
    batch_size=32, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(GTSRB('test'),
    batch_size=1, shuffle=False, **kwargs)
    # valid_loader = torch.utils.data.DataLoader(GTSRB('val'),
    # batch_size=1, shuffle=False, **kwargs)

    return train_loader, test_loader


### RIR data loader
class RIR(Dataset):
    def __init__(self,mode):
        self.data = torch.from_numpy(getattr(readRIR(),'{}_data'.format(mode))).float()##.view(-1,1,-1)
        # print(self.data.shape)
        self.target = torch.from_numpy(getattr(readRIR(),'{}_labels'.format(mode))).long()
        # print(self.target.shape)
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]        
        return x, y
    
    def __len__(self):
        return len(self.data)

class readRIR:
    def __init__(self):
        self.train_data = []
        self.train_labels = []
        self.test_data = []
        self.test_labels = []
        self.validation_data = []
        self.validation_labels = []

        # load train data

        img, lab = load_RIR('data/RIR/train.npy')
        self.train_data.extend(img)
        self.train_labels.extend(lab)

        self.train_data = np.array(self.train_data,dtype=np.float32)
        self.train_labels = np.array(self.train_labels)    

        # load test data

        img, lab = load_RIR('data/RIR/test.npy')
        self.test_data.extend(img)
        self.test_labels.extend(lab)

        self.test_data = np.array(self.test_data,dtype=np.float32)
        self.test_labels = np.array(self.test_labels)   

        # load validation data

        img, lab = load_RIR('data/RIR/valid.npy')
        self.validation_data.extend(img)
        self.validation_labels.extend(lab)

        self.validation_data = np.array(self.validation_data,dtype=np.float32)
        self.validation_labels = np.array(self.validation_labels)  

def load_RIR(fpath):
    images = []
    labels = []
    num_classes = 10
    # with open(fpath, 'rb') as rfile:
    train_dataset =  np.load(fpath)
    images = train_dataset[:,:-1]
    # reshape to 3 channel
    x,y = images.shape
    print(x,y)
    images = images.reshape((x,1,y))
    labels = train_dataset[:,-1:]
    # print("shape is: ",images.shape,labels.shape)
    # for image in train_dataset[:-1]:
    #     # print(image.min(),image.max())
    #     print("image shape: ",image.shape)
    #     images.append(image)
    # for label in train_dataset[-1:]:
    #     # labels.append(np.eye(num_classes)[label])
    #     print("label shape: ",label.shape)
    #     labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels

def load_rir():
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': False}
    train_loader = torch.utils.data.DataLoader(RIR('train'),
    batch_size=64, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(RIR('test'),
    batch_size=1, shuffle=False, **kwargs)
    # valid_loader = torch.utils.data.DataLoader(RIR('val'),
    # batch_size=1, shuffle=False, **kwargs)

    return train_loader, test_loader
