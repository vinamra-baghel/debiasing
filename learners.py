from enum import auto
import gc
from math import fabs
import os
import argparse
from pyexpat import model
import numpy as np
# from symbol import parameters
from copy import deepcopy       ## importing deepcopy
import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from src.helper_functions.helper_functions import mAP, CocoDetection, CutoutPIL, ModelEma, add_weight_decay
from src.models import create_model
from src.loss_functions.losses import AsymmetricLoss, BCEloss
from randaugment import RandAugment
from torch.cuda.amp import GradScaler, autocast
from torch.nn.functional import gumbel_softmax as Gumbel

## bcelearner, similar to asl learner, but with bce loss
class bcelearner(nn.Module):
    def __init__(self, model_train, train_loader, val_loader, args):
        super(bcelearner, self).__init__()
        self.model_train = model_train
        self.model_train_ema = ModelEma(self.model_train, 0.9997)  # 0.9997^641=0.82  # main model

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.args = args

        self.criteria = BCEloss()
        # self.traininfoList = []
        # self.valinfoList = []

        self.type = args.type
        self.model_name = args.model_name

        ## optimizer and scheduler
        self.epoch = 80
        self.stop_epoch = 40
        self.weight_decay = 1e-4
        self.parameters = add_weight_decay(self.model_train, self.weight_decay)
        self.optimizer = torch.optim.Adam(params=self.parameters, lr=args.lr, weight_decay=0)
        self.scheduler = lr_scheduler.OneCycleLR(self.optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=self.epoch, pct_start=0.2)
        self.scaler = GradScaler()

        self.highest_mAP_train = 0
        self.highest_mAP_val = 0

        self.model_train.train()

    def forward(self):
        for epoch in range(self.epoch):
            if epoch>self.stop_epoch:
                break
            
            print("lr: ", self.scheduler.get_last_lr()) 
            self.model_train.train()

            for i, (inputData, target) in enumerate(self.train_loader):
                inputData = inputData.cuda()
                
                target = target.cuda()
                target = target.max(dim=1)[0]
                # print("Target", target.shape)
                with autocast():
                    output_train = self.model_train(inputData).float()
                loss = self.criteria(output_train, target, torch.ones(80).cuda())

                if i%100 == 0 and self.args.print_info == "yes":
                    # printClassLoss(loss, "BCE_Train")
                    print("BCE Train Loss: ", loss)

                loss = loss.sum()

                self.model_train.zero_grad()

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                self.scheduler.step()

                self.model_train_ema.update(self.model_train)

                if i % 100 == 0:
                    # self.traininfoList.append([epoch, i, loss.item()])
                    print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                          .format(epoch, self.epoch, str(i).zfill(3), str(len(self.train_loader)).zfill(3),
                                  self.scheduler.get_last_lr()[0], \
                                  loss.item()))
            
            ## evaluation of the models
            self.model_train.eval()

            for i_val, (inputData_val, target_val) in enumerate(self.val_loader):
                inputData_val = inputData_val.cuda()
                target_val = target_val.cuda()
                target_val = target_val.max(dim=1)[0]
                with autocast():
                    output_val = self.model_train(inputData_val).float()
                loss_val = self.criteria(output_val, target_val, torch.ones(80).cuda())

                if i_val % 100 == 0 and self.args.print_info == "yes":
                    # printClassLoss(loss_val, "BCE_Val")
                    print("BCE Val Loss: ", loss_val)

                loss_val = loss_val.sum()

                if i_val % 100 == 0:
                    # self.valinfoList.append([epoch, i_val, loss_val.item().cpu()])
                    print('Val loss valEpocw [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                          .format(epoch, self.epoch, str(i_val).zfill(3), str(len(self.val_loader)).zfill(3),
                                  self.scheduler.get_last_lr()[0], \
                                  loss_val.item()))

            mAP_score_train = validate_multi(self.train_loader, self.model_train, self.model_train_ema, self.args.print_info)
            mAP_score_val = validate_multi(self.val_loader, self.model_train, self.model_train_ema, self.args.print_info)

            if mAP_score_train > self.highest_mAP_train:
                self.highest_mAP_train = mAP_score_train
            if mAP_score_val > self.highest_mAP_val:
                self.highest_mAP_val = mAP_score_val

            print('Train_data_mAP: current_mAP = {:.2f}, highest_mAP = {:.2f}'.format(mAP_score_train, self.highest_mAP_train))
            print('Val_data_mAP: current_mAP = {:.2f}, highest_mAP = {:.2f}'.format(mAP_score_val, self.highest_mAP_val))

## similar to learner class but with different implementation (hopefully it'll be the correct implementation)
## implementation of this isn't complete yet
class metalearner(nn.Module):
    def __init__(self, model_train, model_val, train_loader, val_loader, args):
        super(metalearner, self).__init__()

        ## models
        self.model_train = model_train
        self.model_train_ema = ModelEma(self.model_train, 0.9997)
        self.model_val = model_val
        self.model_val_ema = ModelEma(self.model_val, 0.9997)

        ## dataloaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        ## args
        self.args = args

        ## training params
        self.lr = args.lr
        self.epochs = 80
        self.stop_epoch = 40
        self.weight_decay = 1e-4

        ## steps per epoch
        self.steps_per_epoch_train = len(train_loader)
        self.steps_per_epoch_val = len(val_loader)

        ## criteria, optimizer and scheduler for model_train and model_val
        self.criteria = BCEloss()

        self.parameters_train = add_weight_decay(self.model_train, self.weight_decay)
        self.optimizer_train = torch.optim.Adam(params=self.parameters_train, lr=self.lr, weight_decay=0)
        self.scheduler_train = lr_scheduler.OneCycleLR(self.optimizer_train, max_lr=self.lr, steps_per_epoch=self.steps_per_epoch_train, epochs=self.epochs, pct_start=0.2)

        self.parameters_val = add_weight_decay(self.model_val, self.weight_decay)
        self.optimizer_val = torch.optim.Adam(params=self.parameters_val, lr=self.lr, weight_decay=0)
        self.scheduler_val = lr_scheduler.OneCycleLR(self.optimizer_val, max_lr=self.lr, steps_per_epoch=self.steps_per_epoch_val, epochs=self.epochs, pct_start=0.2)

        ## scaler
        self.scaler_train = GradScaler()
        self.scaler_val = GradScaler()

        ## highest mAP scores
        self.highest_mAP_train = 0
        self.highest_mAP_val = 0

        ## lists to store training and validation info
        self.trainInfoList = []
        self.valInfoList = []

        ## weights
        self.weights = torch.rand(80).cuda()

    def forward(self):
        print("FORWARD Summing updated weights")

        

        for epoch in range(self.epochs):
            if epoch > self.stop_epoch:
                break
            
            print(self.weights)
            ## training model_train
            for i, (inputData, target) in enumerate(self.train_loader):
                inputData = inputData.cuda()
                target = target.cuda()
                target = target.max(dim=1)[0]
                with autocast():
                    output_train = self.model_train(inputData).float()
                loss = self.criteria(output_train, target, self.weights)
                loss = loss.sum()

                self.model_train.zero_grad()
                self.scaler_train.scale(loss).backward(retain_graph = True)
                self.scaler_train.step(self.optimizer_train)
                self.scaler_train.update()
                self.scheduler_train.step()

                ## model_train ema update
                self.model_train_ema.update(self.model_train)

                if i % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Main Model Weighted Training Loss: {:.1f}'
                          .format(epoch, self.epochs, str(i).zfill(3), str(self.steps_per_epoch_train).zfill(3),
                                  self.scheduler_train.get_last_lr()[0], \
                                  loss.item()))

            ## evaluating model_train
            self.model_train.eval()
            for i_val, (inputData_val, target_val) in enumerate(self.val_loader):
                inputData_val = inputData_val.cuda()
                target_val = target_val.cuda()
                target_val = target_val.max(dim=1)[0]
                with autocast():
                    output_val = self.model_train(inputData_val).float()
                    weight = self.model_val(inputData_val).float()
                weight_ = torch.sigmoid(weight.mean(dim=0))
                loss_val = self.criteria(output_val, target_val, weight_)
                loss_val = loss_val.sum()

                self.model_val.zero_grad()
                self.scaler_val.scale(loss_val).backward(retain_graph = True)
                self.scaler_val.step(self.optimizer_val)
                self.scaler_val.update()
                self.scheduler_val.step()

                ## val model ema update
                self.model_val_ema.update(self.model_val)

                if i_val % 100 == 0:
                    print('Val loss valEpocw [{}/{}], Step [{}/{}], LR {:.1e}, Main Model Weighted Validation Loss: {:.1f}'
                          .format(epoch, self.epochs, str(i_val).zfill(3), str(self.steps_per_epoch_val).zfill(3),
                                  self.scheduler_val.get_last_lr()[0], \
                                  loss_val.item()))

                    loss_val_val = weight.sum()
                    print('Val loss valEpocw [{}/{}], Step [{}/{}], LR {:.1e}, Meta Model weighted Validation Loss: {:.1f}'
                          .format(epoch, self.epochs, str(i_val).zfill(3), str(self.steps_per_epoch_val).zfill(3),
                                  self.scheduler_val.get_last_lr()[0], \
                                  loss_val_val.item()))
                    
                    loss_train_unweighted = self.criteria(output_val, target_val, torch.ones(self.weights.shape).cuda())
                    loss_train_unweighted = loss_train_unweighted.sum()
                    print('Val loss valEpocw [{}/{}], Step [{}/{}], LR {:.1e}, Main Model Unweighted Validation Loss: {:.1f}'
                          .format(epoch, self.epochs, str(i_val).zfill(3), str(self.steps_per_epoch_val).zfill(3),
                                  self.scheduler_val.get_last_lr()[0], \
                                  loss_train_unweighted.item()))

            ## updating weights
            self.weights = weight_

            ## evaluating the models
            self.model_train.eval()
            mAP_score_train = validate_multi(self.train_loader, self.model_train, self.model_train_ema, self.args.print_info)
            mAP_score_val = validate_multi(self.val_loader, self.model_train, self.model_train_ema, self.args.print_info)
            
            self.model_train.train()
            self.model_val.train()

            if mAP_score_train > self.highest_mAP_train:
                self.highest_mAP_train = mAP_score_train
            print('Train_data_mAP: current_mAP = {:.2f}, highest_mAP = {:.2f}'.format(mAP_score_train, self.highest_mAP_train))

            if mAP_score_val > self.highest_mAP_val:
                self.highest_mAP_val = mAP_score_val
            print('Val_data_mAP: current_mAP = {:.2f}, highest_mAP = {:.2f}'.format(mAP_score_val, self.highest_mAP_val))
    

def printClassLoss(loss, name):
    print("--------------------------------")
    for i in range(80):
        print("{} Class_wise {} Loss: {}".format(name, i, loss[i]))