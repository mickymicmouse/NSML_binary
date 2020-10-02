# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 18:48:39 2020

@author: seungjun
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
from efficientnet_pytorch import EfficientNet

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
            test_transform = tv.transforms.Compose([
                    tv.transforms.ToPILImage(mode = 'RGB'),
                    tv.transforms.Resize(512),
                    tv.transforms.CenterCrop(256),
                    tv.transforms.ToTensor(),
                    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                    ])
            batch_dataset = PathDataset(image_path, labels=None, test_mode= True, transform = test_transform)
            batch_loader = DataLoader(dataset=batch_dataset,
                                        batch_size=batch_size,shuffle=False)
            # Train the model 
            for i, images in enumerate(batch_loader):
                y_hat = model(images.to(device)).cpu().numpy()
                y_hat_b = model_b(images.to(device)).cpu().numpy()
                y_hat_r = model_r(images.to(device)).cpu().numpy()
                y= np.argmax(y_hat, axis = 1)
                y_b=np.argmax(y_hat_b, axis = 1)
                y_r=np.argmax(y_hat_r, axis = 1)
                total_result = np.array([y,y_b,y_r])
                before_result = list(np.count_nonzero(total_result, axis=0))
                after_result = [1 if x>=2 else 0 for x in before_result]
                result.extend(after_result)

                

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
        # bgr
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # rgb
        # numpy
        # transforms = need PIL image
                ### REQUIRED: PREPROCESSING ###

        if self.mode:
            
            im = self.transform(im)
            #im = np.array(im)
            #im = im.reshape(3,im.shape[0],im.shape[1])
            return im
        else:
            im = self.transform(im)
            #im = np.array(im)
            #im = im.reshape(3,im.shape[0],im.shape[1])
            
            return im,\
                 torch.tensor(self.labels[index] ,dtype=torch.long)

    def __len__(self): 
        return self.len

class PathDataset_b(Dataset): 
    def __init__(self,image_path, labels=None, test_mode= True, transform = None): 
        self.len = len(image_path)
        self.image_path = image_path
        self.labels = labels 
        self.mode = test_mode
        self.transform = transform

    def __getitem__(self, index): 
        im = cv2.imread(self.image_path[index])
        # bgr
        b,g,r = cv2.split(im)
        zeros = np.zeros((im.shape[0],im.shape[1]), dtype=np.uint8)
        im = cv2.merge((b,zeros,zeros))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # rgb
        # numpy
        # transforms = need PIL image
                ### REQUIRED: PREPROCESSING ###
        
        if self.mode:
            
            im = self.transform(im)
            #im = np.array(im)
            #im = im.reshape(3,im.shape[0],im.shape[1])
            return im
        else:
            im = self.transform(im)
            #im = np.array(im)
            #im = im.reshape(3,im.shape[0],im.shape[1])
            
            return im,\
                 torch.tensor(self.labels[index] ,dtype=torch.long)

    def __len__(self): 
        return self.len

class PathDataset_r(Dataset): 
    def __init__(self,image_path, labels=None, test_mode= True, transform = None): 
        self.len = len(image_path)
        self.image_path = image_path
        self.labels = labels 
        self.mode = test_mode
        self.transform = transform

    def __getitem__(self, index): 
        im = cv2.imread(self.image_path[index])
        # bgr
        b,g,r = cv2.split(im)
        zeros = np.zeros((im.shape[0],im.shape[1]), dtype=np.uint8)
        im = cv2.merge((zeros,zeros,r))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # rgb
        # numpy
        # transforms = need PIL image
                ### REQUIRED: PREPROCESSING ###
        
        if self.mode:
            
            im = self.transform(im)
            #im = np.array(im)
            #im = im.reshape(3,im.shape[0],im.shape[1])
            return im
        else:
            im = self.transform(im)
            #im = np.array(im)
            #im = im.reshape(3,im.shape[0],im.shape[1])
            
            return im,\
                 torch.tensor(self.labels[index] ,dtype=torch.long)

    def __len__(self): 
        return self.len
    
    
def fmeasure(output, target):
    # _, pred = output.topk(1, 1, True, True)
    pred = output.view(-1,1)
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
    args.add_argument('--epoch', type=int, default=2000)
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
    #model = arch.CNN().to(device)
    model_b=EfficientNet.from_pretrained('efficientnet-b3',num_classes=2).cuda()
    model=EfficientNet.from_pretrained('efficientnet-b3',num_classes=2).cuda()
    model_r=EfficientNet.from_pretrained('efficientnet-b3',num_classes=2).cuda()
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_b = torch.optim.Adam(model_b.parameters(), lr=learning_rate)
    optimizer_r = torch.optim.Adam(model_r.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    ############ DONOTCHANGE ###############
    bind_model(model)
    bind_model(model_r)
    bind_model(model_b)
    if config.pause: ## test mode 일때는 여기만 접근
        print('Inferring Start...')
        nsml.paused(scope=locals())
    #######################################

    if config.mode == 'train': ### training mode 일때는 여기만 접근
        print(torch.version.__version__)
        print('Training Start...')

        ############ DONOTCHANGE: Path loader ###############
        root_path = os.path.join(DATASET_PATH,'train')
        image_keys, image_path = path_loader(root_path)
        labels = label_loader(root_path, image_keys)
        ##############################################
        
        
        ##practice##
        #ky = np.array([0,1,2,3,4,5])
        #tt = label_loader('C:\\STbigbase\\pytorch_version',ky)
        #np.count_nonzero(tt)
        count=0
        total_index=[x for x in range(len(labels))]
        true_index=[]
        false_index=[]
        
        for i in range(len(labels)):
            if labels[i]==0:
                true_index.append(labels.index(labels[i],i))
            else:
                false_index.append(labels.index(labels[i],i))

        true_valid_index = true_index[:500]
        false_valid_index = false_index[:500]
        
        
        
        true_train_index = list(set(true_index)-set(true_valid_index))
        false_train_index = list(set(false_index)-set(false_valid_index))
        
        short_ratio = int(len(true_train_index)//len(false_train_index))
        short_add_num = int(len(true_train_index)%len(false_train_index))
        
        false_train_idx = []
        for i in range(short_ratio):
            false_train_idx.extend(false_train_index)
        
        false_train_idx.extend(false_train_index[:short_add_num])
        
        train_index = []
        train_index.extend(true_train_index)
        train_index.extend(false_train_idx)
        valid_index = list(set(true_valid_index) | set(false_valid_index))
        
        #valid data = 1000
        #train_data = 10200
        
        
        #image_path = np.array(['str1','str2','str3','str4','str5','str6'])
        #img1=image_path[total_index]
        #img2=image_path[true_valid_index]
        #img3=np.concatenate([img1,img2])
        labels=np.array(labels)
        
        
        ###
        """
        total_len = len(image_path)
        train_ratio = 0.8
        train_image = image_path[:int(total_len*train_ratio)]
        train_label = labels[:int(total_len*train_ratio)]
        valid_image = image_path[int(total_len*train_ratio):]
        valid_label = labels[int(total_len*train_ratio):]
        """
        
        train_image = image_path[train_index]
        train_label = labels[train_index]
        valid_image = image_path[valid_index]
        valid_label = labels[valid_index]
        
        print('number of train image is : '+str(len(train_image)))
        print('number of valid image is : '+str(len(valid_image)))
        print('number of train true is : '+str(len(true_train_index)))
        print('number of train false is : '+str(len(false_train_idx)))        
        
        
        
        train_transform1 = tv.transforms.Compose([
                tv.transforms.ToPILImage(mode = 'RGB'),
                tv.transforms.Resize(512),
                tv.transforms.RandomAffine(0.1),
                tv.transforms.RandomCrop(256),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

                ])
    
        # 회전
        train_transform2 = tv.transforms.Compose([
                tv.transforms.ToPILImage(mode = 'RGB'),
                tv.transforms.Resize(512),
                tv.transforms.RandomCrop(256),
                tv.transforms.RandomHorizontalFlip(p=0.5),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                ])
        #수평 플립
        train_transform3 = tv.transforms.Compose([
                tv.transforms.ToPILImage(mode = 'RGB'),
                tv.transforms.Resize(512),
                tv.transforms.RandomCrop(256),
                tv.transforms.RandomVerticalFlip(p=0.5),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                ])
        #수직 플립
        train_transform4 = tv.transforms.Compose([
                tv.transforms.ToPILImage(mode = 'RGB'),
                tv.transforms.Resize(512),
                tv.transforms.RandomCrop(256),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                ])
        # original
        
        test_transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(mode = 'RGB'),
                tv.transforms.Resize(512),
                tv.transforms.CenterCrop(256),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                ])
        
        train_data1 = PathDataset(train_image, train_label, test_mode=False,transform = train_transform1)
        train_data2 = PathDataset(train_image, train_label, test_mode=False,transform = train_transform2)
        train_data3 = PathDataset(train_image, train_label, test_mode=False,transform = train_transform3)
        train_data4 = PathDataset(train_image, train_label, test_mode=False,transform = train_transform4)
        
        train_data1_b = PathDataset_b(train_image, train_label, test_mode=False,transform = train_transform1)
        train_data2_b = PathDataset_b(train_image, train_label, test_mode=False,transform = train_transform2)
        train_data3_b = PathDataset_b(train_image, train_label, test_mode=False,transform = train_transform3)
        train_data4_b = PathDataset_b(train_image, train_label, test_mode=False,transform = train_transform4)
        
        train_data1_r = PathDataset_r(train_image, train_label, test_mode=False,transform = train_transform1)
        train_data2_r = PathDataset_r(train_image, train_label, test_mode=False,transform = train_transform2)
        train_data3_r = PathDataset_r(train_image, train_label, test_mode=False,transform = train_transform3)
        train_data4_r = PathDataset_r(train_image, train_label, test_mode=False,transform = train_transform4)

        
        valid_data = PathDataset(valid_image, valid_label, test_mode=False,transform = test_transform)
        valid_data_b = PathDataset_b(valid_image, valid_label, test_mode=False,transform = test_transform)
        valid_data_r = PathDataset_r(valid_image, valid_label, test_mode=False,transform = test_transform)
        
        
        train_loader = DataLoader(dataset=ConcatDataset([train_data1,train_data2,train_data3,train_data4]), 
                batch_size=batch_size, shuffle=True, drop_last = True)
        train_loader_b = DataLoader(dataset=ConcatDataset([train_data1_b,train_data2_b,train_data3_b,train_data4_b]), 
                batch_size=batch_size, shuffle=True, drop_last = True)
        train_loader_r = DataLoader(dataset=ConcatDataset([train_data1_r,train_data2_r,train_data3_r,train_data4_r]), 
                batch_size=batch_size, shuffle=True, drop_last = True)
     
        
        valid_loader = DataLoader(dataset=valid_data, 
                batch_size=batch_size, shuffle=False, drop_last = True)
        valid_loader_b = DataLoader(dataset=valid_data_b, 
                batch_size=batch_size, shuffle=False, drop_last = True)
        valid_loader_r = DataLoader(dataset=valid_data_r, 
                batch_size=batch_size, shuffle=False, drop_last = True)
        
                
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
            answer=[]
            y=[]
            y_b=[]
            y_r=[]
            
            
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

                
            for i, (images, labels) in enumerate(train_loader_r):


                model_r.train()
                
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass

                outputs_r = model_r(images)

                loss_r = criterion(outputs_r, labels)
                
                # Backward and optimize

                optimizer_r.zero_grad()
                

                loss_r.backward()
                

                optimizer_r.step()
    
            for i, (images, labels) in enumerate(train_loader_b):

                model_b.train()

                
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass

                outputs_b = model_b(images)

                loss_b = criterion(outputs_b, labels)

                
                # Backward and optimize

                optimizer_b.zero_grad()

                loss_b.backward()

                optimizer_b.step()

                

            for j, (vimages, vlabels) in enumerate(valid_loader):
                model.eval()

                
                vimages = vimages.to(device)
                vlabels = vlabels.to(device)
                
                voutputs = model(vimages)

                
                #crossentropy = criterion(voutputs, vlabels)
                
                
                

                voutputs = voutputs.cpu().detach().numpy()
                answer.extend(vlabels.cpu().tolist())
                

                y.extend(np.argmax(voutputs, axis = 1).tolist())

                
            for j, (vimages, vlabels) in enumerate(valid_loader_b):

                model_b.eval()
                
                vimages = vimages.to(device)
                vlabels = vlabels.to(device)
                

                voutputs_b = model_b(vimages)

                
                #crossentropy = criterion(voutputs, vlabels)
                
                
                

                voutputs_b = voutputs_b.cpu().detach().numpy()
                

                y_b.extend(np.argmax(voutputs_b, axis = 1).tolist())

                
            for j, (vimages, vlabels) in enumerate(valid_loader_r):

                model_r.eval()

                
                vimages = vimages.to(device)
                vlabels = vlabels.to(device)
                
                voutputs_r = model_r(vimages)
                
                #crossentropy = criterion(voutputs, vlabels)
                
                
                
                voutputs_r = voutputs_r.cpu().detach().numpy()

                y_r.extend(np.argmax(voutputs_r, axis = 1).tolist())
                                
                
                
            total_result = np.array([y,y_b,y_r])
            before_result = list(np.count_nonzero(total_result, axis=0))
            after_result = [1 if x>=2 else 0 for x in before_result]
            
            
            TP,FP,TN,FN,pred_len, gt_len=fmeasure(torch.tensor(after_result),torch.tensor(answer))
            
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
            f1= (2*prec*sens / (prec + sens))
            total_metric=(f1+accuracy+sens+spec+prec+npv)/6
            
            if best<total_metric:
                best=total_metric
                print('{0}/{1}'.format(best,total_metric))
                print('acc : {} sens : {} spec :{} prec :{} npv :{}  f1 :{}'.format(accuracy,sens,spec,prec,npv,f1))
                nsml.report(summary=True, step=epoch, epoch_total=num_epochs, loss=loss.item())#, acc=train_acc)
                nsml.save('best')

                

            
            nsml.report(summary=True, step=epoch, epoch_total=num_epochs, loss=loss.item(), acc = total_metric)#, acc=train_acc)
            print(total_metric)
            nsml.save(epoch)
            

            
        
