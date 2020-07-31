import matplotlib.pyplot as plt
from os.path import isfile
import torch.nn.init as init
import torch
import torch.nn as nn
import numpy as np
import pandas as pd 
import os
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import Dataset
from torchvision import transforms
from torch.optim import Adam, SGD, RMSprop
import time
from torch.autograd import Variable
import torch.functional as F
from tqdm import tqdm
from sklearn import metrics
import urllib
import pickle
import cv2
import torch.nn.functional as F
from torchvision import models
import random
from apex import amp
import sys
from efficientnet_pytorch import EfficientNet
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue, IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RGBShift, Normalize, IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, JpegCompression, MultiplicativeNoise, Downscale)
from apex.parallel import DistributedDataParallel as DDP
from apex.parallel import convert_syncbn_model, SyncBatchNorm
from apex.fp16_utils import *
from apex.multi_tensor_apply import multi_tensor_applier
import argparse
from random import sample
from tensorboardX import SummaryWriter 
from collections import OrderedDict
from torch.optim.lr_scheduler import _LRScheduler
writer = SummaryWriter()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['MASTER_ADDR'] = '127.0.0.1'
parse = argparse.ArgumentParser()
parse.add_argument("--local_rank", type=int, default=0) 
args = parse.parse_args()
local_rank = args.local_rank
torch.cuda.set_device(local_rank)
torch.distributed.init_process_group(backend="nccl")
device = torch.device("cuda:{}".format(local_rank))

print('local_rank',local_rank)

num_classes = 2

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
#seed_everything(1234)

val_csv1 = pd.read_csv('/hdd1/faceforensic/test2/val.csv')
val_csv = val_csv1.sample(n = 50000)
train_csv1 = pd.read_csv('/hdd1/dfdc_train/train/train.csv')
train_csv = train_csv1.sample(n = 80000)
p = train_csv.iloc[:,:1].values
p1 = p.tolist()
label = train_csv.iloc[:,1:].values.ravel()
label1 = label.tolist()
for i in range(len(label1)):
    p1[i].append(label1[i])           
    p2 = np.array(p1)
    train_real=[]
    train_fake=[]
    
r = 0
f = 0
for i in range(len(p2)):
    if p2[i][1] == str(1):
        train_real.append(p2[i][:])
        r= r+1
    else:
        train_fake.append(p2[i][:])
        f = f+1

print('r',r)
print('f',f)
train_real1 = np.array(train_real,dtype='str')
train_fake = sample(train_fake,len(train_real1))
train_fake = np.array(train_fake,dtype='str')        
train_data = np.concatenate((train_real1,train_fake),axis=0)
np.random.shuffle(train_data)

class PolyLR(_LRScheduler):
    """Set the learning rate of each parameter group to the initial lr decayed
    by gamma every epoch. When last_epoch=-1, sets initial lr as lr.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, max_iter, power=0.9, last_epoch=-1):
        self.max_iter = max_iter
        self.power = power
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (1 - self.last_epoch/self.max_iter) ** self.power
                for base_lr in self.base_lrs]


class NMTCritierion(nn.Module):
    """
    TODO:
    1. Add label smoothing
    """
    def __init__(self, label_smoothing=0.0):
        super(NMTCritierion, self).__init__()
        self.label_smoothing = label_smoothing
        self.LogSoftmax = nn.LogSoftmax()

        if label_smoothing > 0:
            self.criterion = nn.KLDivLoss(size_average=False)
        else:
            self.criterion = nn.NLLLoss(size_average=False, ignore_index=100000)
        self.confidence = 1.0 - label_smoothing

    def _smooth_label(self, num_tokens):
        # When label smoothing is turned on,
        # KL-divergence between q_{smoothed ground truth prob.}(w)
        # and p_{prob. computed by model}(w) is minimized.
        # If label smoothing value is set to zero, the loss
        # is equivalent to NLLLoss or CrossEntropyLoss.
        # All non-true labels are uniformly set to low-confidence.
        one_hot = torch.randn(1, num_tokens)
        one_hot.fill_(self.label_smoothing / (num_tokens - 1))
        return one_hot

    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def forward(self, dec_outs, labels):
        scores = self.LogSoftmax(dec_outs)
        num_tokens = scores.size(-1)

        # conduct label_smoothing module
        gtruth = labels.view(-1)
        if self.confidence < 1:
            tdata = gtruth.detach()
            one_hot = self._smooth_label(num_tokens)  # Do label smoothing, shape is [M]
            if labels.is_cuda:
                one_hot = one_hot.cuda()
            tmp_ = one_hot.repeat(gtruth.size(0), 1)  # [N, M]
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)  # after tdata.unsqueeze(1) , tdata shape is [N,1]
            gtruth = tmp_.detach()
        loss = self.criterion(scores, gtruth)
        return loss

class MyDataset(Dataset):
    
    def __init__(self, data, transform=None):
        self.df = data
        self.transform = transform
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        label1 = int(self.df[idx][1])
        c = str(self.df[idx][0])
        image = cv2.imread(c)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = RandomRotate90()(image=image)['image']
        image = Flip()(image=image)['image']
        image = JpegCompression(quality_lower=9, quality_upper=10)(image=image)['image']
        image = Transpose()(image=image)['image']
        image = Downscale()(image=image)['image']
        image = IAAAdditiveGaussianNoise()(image=image)['image']
        image = Blur(blur_limit=7)(image=image)['image']
        image = ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45)(image=image)['image']
        image = IAAPiecewiseAffine()(image=image)['image']
        image = RGBShift()(image=image)['image']
        image = RandomBrightnessContrast()(image=image)['image']
        image = HueSaturationValue()(image=image)['image']
        image = transforms.ToPILImage()(image)
        if self.transform:
            image = self.transform(image)
        return image, label1

class MyDataset_val(Dataset):
    
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        p_val = val_csv.iloc[:,:1].values
        label_val = val_csv.iloc[:,1:].values.ravel()
        label = int(label_val[idx])
        c = str(p_val[idx])[2:-2]
        image = cv2.imread(c)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transforms.ToPILImage()(image)
        if self.transform:
            image = self.transform(image)
        
        return image, label

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation((-120, 120)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

batch_size = 16
n_epochs = 35
lr = 0.01

trainset = MyDataset(train_data, transform =train_transform)                             
train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
train_loader = torch.utils.data.DataLoader(trainset, sampler=train_sampler, batch_size=batch_size, num_workers=2)
valset = MyDataset_val(val_csv, transform =train_transform)
val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, num_workers=2)
model = EfficientNet.from_pretrained('efficientnet-b5')
in_features = model._fc.in_features
model._fc = nn.Linear(in_features, num_classes)
model = convert_syncbn_model(model)
model = model.cuda()
#optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay= 1e-4)
model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)
optimizer.load_state_dict(torch.load('dataset-dfdc-val-deepfakedetection-mixed-val_loss-0.6705.pth')['optimizer'])
amp.load_state_dict(torch.load('dataset-dfdc-val-deepfakedetection-mixed-val_loss-0.6705.pth')['amp'])
model.load_state_dict(torch.load('dataset-dfdc-val-deepfakedetection-mixed-val_loss-0.6705.pth')['model'])
model = DDP(model)
#criterion = nn.CrossEntropyLoss()
criterion = NMTCritierion(0.1)
scheduler = PolyLR(optimizer, max_iter = n_epochs)

def train_model(epoch):
    model.train() 
    avg_loss = 0.
    accuracy = 0.
    pred = []
    true = []
    optimizer.zero_grad()
    for idx, (imgs, labels) in enumerate(train_loader):
        labels1 = np.array(list(labels))
        labels2 = torch.from_numpy(labels1)
        labels_train1 = Variable(labels2)
        imgs_train, labels_train = imgs.to(device), labels_train1.to(device)
        output_train = model(imgs_train)
        loss = criterion(output_train,labels_train)
        _, preds = torch.max(output_train, 1)       
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step() 
        optimizer.zero_grad() 
        avg_loss += loss.item() / len(train_loader)
        preds = preds.cpu().numpy()
        target = labels_train1.cpu().numpy()
        preds = np.reshape(preds,(len(preds),1))
        target = np.reshape(target,(len(preds),1))
        
        for i in range(len(preds)):
            pred.append(preds[i])
            true.append(target[i])

    writer.add_scalar('avg_loss', avg_loss,epoch)
    return avg_loss, pred, true

def test_model():
    avg_val_loss = 0.
    val_accuracy = 0.
    model.eval()
    pred = []
    true = []    
    with torch.no_grad():
        for idx, (imgs, labels) in enumerate(val_loader):
            labels3 = np.array(list(labels))
            labels4 = torch.from_numpy(labels3)
            labels_valid1 = Variable(labels4)
            imgs_valid, labels_valid = imgs.to(device), labels_valid1.to(device)
            output_test = model(imgs_valid)
            avg_val_loss += criterion(output_test, labels_valid).item() / len(val_loader)
            _, preds = torch.max(output_test, 1)
            preds = preds.cpu().numpy()
            target = labels_valid1.cpu().numpy()
            preds = np.reshape(preds,(len(preds),1))
            target = np.reshape(target,(len(preds),1))
        
            for i in range(len(preds)):
                pred.append(preds[i])
                true.append(target[i])
            
    writer.add_scalar('avg_val_loss', avg_val_loss,epoch)        
    return avg_val_loss, pred, true
    
best_avg_loss = 100.0

def performance_matrix(true,pred):
    precision = metrics.precision_score(true,pred,average='macro')
    recall = metrics.recall_score(true,pred,average='macro')
    accuracy = metrics.accuracy_score(true,pred)
    writer.add_scalar('precision', precision, epoch)
    writer.add_scalar('recall', recall, epoch)
    writer.add_scalar('accuracy', accuracy, epoch)
  #  f1_score = metrics.f1_score(true,pred,average='macro')
    print('Precision: {} Recall: {}, Accuracy: {}'.format(precision*100,recall*100,accuracy*100))
def val_performance_matrix(true,pred):
    precision = metrics.precision_score(true,pred,average='macro')
    recall = metrics.recall_score(true,pred,average='macro')
    accuracy = metrics.accuracy_score(true,pred)
    writer.add_scalar('val_precision', precision, epoch)
    writer.add_scalar('val_recall', recall, epoch)
    writer.add_scalar('val_accuracy', accuracy, epoch)
  #  f1_score = metrics.f1_score(true,pred,average='macro')
    print('val_Precision: {} val_Recall: {}, val_Accuracy: {}'.format(precision*100,recall*100,accuracy*100))
    
for epoch in range(n_epochs):
   # print('lr:', optimizer.get_lr()[0]) 
    start_time = time.time()
    avg_loss, pred, true = train_model(epoch)
    avg_val_loss, val_pred, val_true = test_model()
    elapsed_time = time.time() - start_time 
    print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(
        epoch + 1, n_epochs, avg_loss, avg_val_loss, elapsed_time))
    performance_matrix(true,pred)
    val_performance_matrix(val_true, val_pred)
    if avg_val_loss < best_avg_loss:
        best_avg_loss = avg_val_loss
        print('best_avg_loss:',best_avg_loss)
        if local_rank == 0:
            checkpoint = {'model': model.module.state_dict(),'optimizer': optimizer.state_dict(),'amp': amp.state_dict()}
            torch.save(checkpoint, 'train.pth')
    scheduler.step()
