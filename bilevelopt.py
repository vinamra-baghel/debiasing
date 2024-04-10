import gc
import os
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
from torch.optim import lr_scheduler
from torch.cuda.amp import GradScaler, autocast
from torch.nn.functional import gumbel_softmax as Gumbel
from torch.nn import BCEWithLogitsLoss, BCELoss
from src.helper_functions.helper_functions import mAP, ModelEma, add_weight_decay
from src.loss_functions.losses import AsymmetricLoss, BCEloss

from cocostats import coco2014
from learner import getInverseClassFreqs, validate_multi

class lambdaModel(nn.Module):
    def __init__(self, n_classes):
        super(lambdaModel, self).__init__()
        # self.weights = nn.Parameter(torch.rand(n_classes).cuda(), requires_grad=True)
        self.fc = nn.Linear(n_classes, 1, bias=False)  # Linear layer with 80 input features and 1 output

    def forward(self, x):
        # return self.weights
        return self.fc(x)

class BiLevelOpt(nn.Module):
    def __init__(self, model_train, model_val, train_loader, val_loader, args):
        super(BiLevelOpt, self).__init__()
        self.model_train = model_train
        self.model_train_ema = ModelEma(self.model_train, 0.9997)  # 0.9997^641=0.82  # main model
        self.model_val = model_val
        self.model_val_ema = ModelEma(self.model_val, 0.9997) # 0.9997^641=0.82  # meta model @manoj : please change the nomenclature => too confusing

        self.type = args.type
        self.model_name = args.model_name

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args

        self.lr = args.lr
        self.meta_lr = args.meta_lr
        self.epochs = 80
        self.stop_epoch = 40    ## can be updated during training process
        self.weight_decay = 1e-4

        self.steps_per_epoch_train = len(train_loader)
        self.steps_per_epoch_val = len(val_loader)

        self.optimizer_train = torch.optim.Adam(params=self.model_train.parameters(), lr=self.lr, weight_decay=0)  # optimizer for main model
        self.optimizer_val = torch.optim.Adam(params=self.model_val.parameters(), lr=self.meta_lr, weight_decay=0) # optimizer for meta model

        self.highest_mAP_train = 0
        self.highest_mAP_val = 0

        self.criteria = BCEloss()
        self.criteria2 = BCEWithLogitsLoss(reduction='none')
        self.trainInfoList = []
        self.modeltrainvalinfo = []
        self.valInfoList = []

        if args.type == 'objpinv':
            self.weights = getInverseClassFreqs()
        else:
            self.weights = lambdaModel(80).cuda()

        self.model_train.train()
        self.model_val.train()
    
    def loss(self, y_pred, target, inner_opt=True):
        if inner_opt:
            # Inner optimization: fix weights, update resnet parameters
            # with torch.no_grad():
                # y_pred = self.resnet(x)
            losses = [self.criteria(y_pred[:, i], target[:, i]) for i in range(self.k)]
            inner_loss = self.weights * torch.stack(losses)
            inner_loss = inner_loss.sum()
            return inner_loss
        else:
            # Outer optimization: fix resnet parameters, update weights
            # with torch.no_grad():
                # y_pred = self.resnet(x)  # Don't update resnet in outer loop
            losses = [self.criteria(y_pred[:, i], target[:, i]) for i in range(self.k)]
            outer_loss = self.weights * torch.stack(losses)
            outer_loss = outer_loss.sum()
            return outer_loss
    
    def forward(self):
        print("FORWARD SUMMING")
        for epoch in range(self.epochs):
            print(f"Lambda Weights {self.weights}")
            if epoch > self.stop_epoch:
                break
            
            # inner model
            fasttrain = deepcopy(self.model_train)
            parameters_fasttrain = add_weight_decay(fasttrain, self.weight_decay)
            optimizer_fasttrain = torch.optim.Adam(params=parameters_fasttrain, lr=self.lr, weight_decay=0)
            fasttrain.cuda()
            fasttrain.train()

            for i, (inputData, target) in enumerate(self.train_loader):
                inputData = inputData.cuda()
                target = target.cuda()
                target = target.max(dim=1)[0]
                with autocast():
                    output_train = fasttrain(inputData).float()
                loss = self.criteria(output_train, target)

                if i % 100 == 0 and self.args.print_info == "yes":
                    print("Sum Train Loss: ", loss)

                loss = self.weights(loss)
                # print(f"Loss shape {loss.shape}")
                loss = loss.sum()              # weighted sum over losses $\lambda*L_{i}$
                # print(f"Loss {loss}")

                optimizer_fasttrain.zero_grad()
                loss.backward(retain_graph=True)
                optimizer_fasttrain.step()

                # for name, param in fasttrain.named_parameters():
                #     if param.grad is not None:
                #         print(name)

                if i % 100 == 0:
                    # self.trainInfoList.append([epoch, i, loss.item()])
                    print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Main Model Weighted Training Loss: {:.1f}'
                          .format(epoch, self.epochs, str(i).zfill(3), str(self.steps_per_epoch_train).zfill(3),
                                  self.lr, \
                                  loss.item()))
                    # torch.save(fasttrain.state_dict(), os.path.join(
                    # 'models/{}/{}/models_train/'.format(self.type, self.model_name), 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))
            
            with torch.no_grad():        
                self.optimizer_train.zero_grad()
                self.model_train.load_state_dict(fasttrain.state_dict())

            # ema model update
            self.model_train_ema.update(self.model_train)

            self.model_train.eval()
            fasttrain.eval()

            # outer model
            outerModel = deepcopy(self.weights)
            outerModel.requires_grad_(True)
            parameters_outerModel = add_weight_decay(outerModel, self.weight_decay)
            optimizer_outerModel = torch.optim.Adam(params=parameters_outerModel, lr=self.lr, weight_decay=0)
            outerModel.cuda()
            outerModel.train()

            print(f"Lambda Weights {self.weights}")
            for i_val, (inputData_val, target_val) in enumerate(self.val_loader):
                inputData_val = inputData_val.cuda()
                target_val = target_val.cuda()
                target_val = target_val.max(dim=1)[0]

                with autocast():
                    # output_val = fasttrain(inputData_val).float()
                    output_val = self.model_train(inputData_val).float()
                loss_val = self.criteria(output_val, target_val) # validation loss
                
                if i_val % 100 == 0 and self.args.print_info == "yes":
                    print("Sum_Val Meta Model: ", loss_val)

                loss_val = self.weights(loss_val) # number_classes
                loss_val = loss_val.sum()

                optimizer_outerModel.zero_grad()
                loss_val.backward(retain_graph=True)
                optimizer_outerModel.step()

            with torch.no_grad():        
                self.optimizer_val.zero_grad()
                self.model_val.load_state_dict(outerModel.state_dict())
            
            # ema model update
            self.model_val_ema.update(self.model_val)

            self.model_val.eval()

            del fasttrain, parameters_fasttrain, optimizer_fasttrain, outerModel, parameters_outerModel, optimizer_outerModel
            torch.cuda.empty_cache()
            gc.collect()
            
            print("---------evaluating the models---------\n")
            self.model_train.eval()
            self.model_val.eval()
            mAP_score_train = validate_multi(self.train_loader, self.model_train, self.model_train_ema, self.args.print_info)
            mAP_score_val = validate_multi(self.val_loader, self.model_train, self.model_train_ema, self.args.print_info)
            self.model_train.train()
            self.model_val.train()
            if mAP_score_train > self.highest_mAP_train:
                self.highest_mAP_train = mAP_score_train
                # torch.save(self.model_train.state_dict(), os.path.join(
                #     'models/{}/{}/models_train/'.format(self.type, self.model_name), 'model-highest.ckpt'))
            print('Train_data_mAP: current_mAP = {:.2f}, highest_mAP = {:.2f}'.format(mAP_score_train, self.highest_mAP_train))

            if mAP_score_val > self.highest_mAP_val:
                self.highest_mAP_val = mAP_score_val
                # torch.save(self.model_val.state_dict(), os.path.join(
                    # 'models/{}/{}/models_val/'.format(self.type, self.model_name), 'model-highest.ckpt'))
            print('Val_data_mAP: current_mAP = {:.2f}, highest_mAP = {:.2f}'.format(mAP_score_val, self.highest_mAP_val))