#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 00:53:01 2020

@author: hihyun
"""

import os
import argparse
import sys
import time
import arch
import cv2 
import numpy as np
import nsml
from nsml.constants import DATASET_PATH, GPU_NUM
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.metrics import *
import torchvision as tv

######################## DONOTCHANGE ###########################
def bind_model(model):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        torch.save(model.state_dict(),os.path.join(dir_name, 'model'))
        print('model saved!')

    def load(dir_name):
        model.load_state_dict(torch.load(os.path.join(dir_name, 'model')))
        model.eval()
        print('model loaded!')

    def infer(image_path):

        result = []
        with torch.no_grad():             
            batch_loader = DataLoader(dataset=PathDataset(image_path, labels=None),
                                        batch_size=1,shuffle=False)
            # Train the model 
            for i, images in enumerate(batch_loader):
                y_hat = model(images.to(device)).cpu().numpy()
                result.extend(np.argmax(y_hat, axis=1))

        print('predicted')
        return np.array(result)

    nsml.bind(save=save, load=load, infer=infer)


def path_loader (root_path):
    image_path = []
    image_keys = []
    for _,_,files in os.walk(os.path.join(root_path,'train_data')):
        for f in files:
            path = os.path.join(root_path,'train_data',f)
            if path.endswith('.png'):
                image_keys.append(int(f[:-4]))
                image_path.append(path)

    return np.array(image_keys), np.array(image_path)


def label_loader (root_path, keys):
    labels_dict = {}
    labels = []
    with open (os.path.join(root_path,'train_label'), 'rt') as f :
        for row in f:
            row = row.split()
            labels_dict[int(row[0])] = (int(row[1]))
    for key in keys:
        labels = [labels_dict[x] for x in keys]
    return labels
############################################################


class PathDataset(Dataset): 
    def __init__(self,image_path, labels=None, test_mode= True, transform = None): 
        self.len = len(image_path)
        self.image_path = image_path
        self.labels = labels 
        self.mode = test_mode
        self.transform = transform

    def __getitem__(self, index): 
        im = cv2.imread(self.image_path[index])
        # numpy
        # transforms = need PIL image
                ### REQUIRED: PREPROCESSING ###

        
        
        if self.mode:

            im = im.reshape(3,im.shape[0],im.shape[1])
            return torch.tensor(im,dtype=torch.float32)
        else:
            im = self.transform(im)
            im = np.array(im)
            im = im.reshape(3,im.shape[0],im.shape[1])
            
            return torch.tensor(im,dtype=torch.float32),\
                 torch.tensor(self.labels[index] ,dtype=torch.long)

    def __len__(self): 
        return self.len

    
    
def fmeasure(output, target):
    _, pred = output.topk(1, 1, True, True)
    pred = pred.view(-1,1)
    target = target.view(-1,1)

    #overlap = ((pred== 1) + (target == 1)).gt(1)
    #overlap = overlap.view(-1,1)
    TP = len(np.where((pred==1)&(target==1)==True)[0]) # True positive
    FP = len(np.where((pred==1)&(target==0)==True)[0]) # Condition positive = TP + FN
    TN = len(np.where((pred==0)&(target==0)==True)[0])
    FN = len(np.where((pred==0)&(target==1)==True)[0])


    #overlap_len = overlap.data.long().sum()
    pred_len = pred.data.long().sum()
    gt_len   =  target.data.long().sum()

    return TP,FP,TN,FN,pred_len, gt_len


if __name__ == '__main__':

    ########## ENVIRONMENT SETUP ############
    args = argparse.ArgumentParser()

    ########### DONOTCHANGE: They are reserved for nsml ###################
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    ######################################################################

    # hyperparameters
    args.add_argument('--epoch', type=int, default=100)
    args.add_argument('--batch_size', type=int, default=64) 
    args.add_argument('--learning_rate', type=int, default=0.0001)
    args.add_argument('--train_ratio', type=int, default=0.8)

    config = args.parse_args()

    # training parameters
    num_epochs = config.epoch
    batch_size = config.batch_size
    num_classes = 2
    learning_rate = config.learning_rate 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # model setting ## 반드시 이 위치에서 로드해야함
    model = arch.CNN().to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    ############ DONOTCHANGE ###############
    bind_model(model)
    if config.pause: ## test mode 일때는 여기만 접근
        print('Inferring Start...')
        nsml.paused(scope=locals())
    #######################################

    if config.mode == 'train': ### training mode 일때는 여기만 접근
        print('Training Start...')
        nsml.load(checkpoint='36',session='KHD009/Breast_Pathology/34')
        nsml.save('saved')
        exit()

        ############ DONOTCHANGE: Path loader ###############
        root_path = os.path.join(DATASET_PATH,'train')
        image_keys, image_path = path_loader(root_path)
        labels = label_loader(root_path, image_keys)
        ##############################################
        total_len = len(image_path)
        train_ratio = 0.8
        train_image = image_path[:int(total_len*train_ratio)]
        train_label = labels[:int(total_len*train_ratio)]
        valid_image = image_path[int(total_len*train_ratio):]
        valid_label = labels[int(total_len*train_ratio):]
        
        train_transform1 = tv.transforms.Compose([
                tv.transforms.ToPILImage(mode = 'RGB'),
                tv.transforms.Resize(512),
                tv.transforms.RandomAffine(0.1),
                tv.transforms.RandomCrop(64),

                ])
    
        # 회전
        train_transform2 = tv.transforms.Compose([
                tv.transforms.ToPILImage(mode = 'RGB'),
                tv.transforms.Resize(512),
                tv.transforms.RandomCrop(64),
                tv.transforms.RandomHorizontalFlip(p=0.5),

                ])
        #수평 플립
        train_transform3 = tv.transforms.Compose([
                tv.transforms.ToPILImage(mode = 'RGB'),
                tv.transforms.Resize(512),
                tv.transforms.RandomCrop(64),
                tv.transforms.RandomVerticalFlip(p=0.5)
                ])
        #수직 플립
        train_transform4 = tv.transforms.Compose([
                tv.transforms.ToPILImage(mode = 'RGB'),
                tv.transforms.Resize(512),
                tv.transforms.RandomCrop(64),
                ])
        # original
        
        test_transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(mode = 'RGB'),
                tv.transforms.Resize(512),
                tv.transforms.CenterCrop(64)
                ])
        
        train_data1 = PathDataset(train_image, train_label, test_mode=False,transform = train_transform1)
        train_data2 = PathDataset(train_image, train_label, test_mode=False,transform = train_transform2)
        train_data3 = PathDataset(train_image, train_label, test_mode=False,transform = train_transform3)
        train_data4 = PathDataset(train_image, train_label, test_mode=False,transform = train_transform4)
        
        
        valid_data = PathDataset(valid_image, valid_label, test_mode=False,transform = test_transform)
        train_loader = DataLoader(dataset=ConcatDataset(train_data1,train_data2,train_data3,train_data4), 
                batch_size=batch_size, shuffle=True, drop_last = True)
        valid_loader = DataLoader(dataset=valid_data, 
                batch_size=batch_size, shuffle=True, drop_last = True)
                
        print('train number is : '+str(len(train_data1)))
        print('valid number is : '+str(len(valid_data)))
        best=0 #for metric
        
        # Train the model
        for epoch in range(num_epochs):
            pred_sum = 0#model output
            gt_sum = 0#label
            tp_sum=0.00001
            fp_sum=0.00001
            fn_sum=0.00001
            tn_sum=0.00001
            acc=0
            total=0
            for i, (images, labels) in enumerate(train_loader):
                model.train()
                
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            
            for j, (vimages, vlabels) in enumerate(valid_loader):
                model.eval()
                
                vimages = vimages.to(device)
                vlabels = vlabels.to(device)
                
                voutputs = model(vimages)
                crossentropy = criterion(voutputs, vlabels)
                TP,FP,TN,FN,pred_len, gt_len=fmeasure(voutputs.cpu(),vlabels.cpu())

                voutputs = voutputs.cpu().detach().numpy()
                vlabels = vlabels.cpu().detach().numpy()
                
                tp_sum += TP
                fp_sum += FP
                fn_sum += FN
                tn_sum += TN
                pred_sum += pred_len
                gt_sum += gt_len
                acc=acc+TP+TN
                total+=len(voutputs)
            
            
            #metric 통합
            
            accuracy=acc/total
            sens=tp_sum/(tp_sum+fn_sum)
            spec=tn_sum/(tn_sum+fp_sum)
            prec=tp_sum/(tp_sum+fp_sum)
            npv=tn_sum/(tn_sum+fn_sum)
            f1= (2*prec*sens / (prec + sens)) * 100
            total_metric=(f1+accuracy+sens+spec+prec+npv)/6
            
            if best<total_metric:
                best=total_metric
                nsml.report(summary=True, step=epoch, epoch_total=num_epochs, loss=loss.item())#, acc=train_acc)
                nsml.save('best')

                

            
            nsml.report(summary=True, step=epoch, epoch_total=num_epochs, loss=loss.item(), acc = total_metric)#, acc=train_acc)
            print(total_metric)
            nsml.save(epoch)
            

            
        
