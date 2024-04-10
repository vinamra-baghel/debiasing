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
from src.helper_functions.helper_functions import mAP, ModelEma, add_weight_decay
from src.loss_functions.losses import AsymmetricLoss, BCEloss

from cocostats import coco2014

def train_multi_label_coco(model, train_loader, val_loader, lr):
    ema = ModelEma(model, 0.9997)  # 0.9997^641=0.82

    # set optimizer
    Epochs = 80
    Stop_epoch = 40
    weight_decay = 1e-4
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    parameters = add_weight_decay(model, weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, valEpocws=Epochs,
                                        pct_start=0.2)

    highest_mAP = 0
    trainInfoList = []
    scaler = GradScaler()
    for epoch in range(Epochs):
        if epoch > Stop_epoch:
            break
        # print("lr ", scheduler.get_last_lr())
        for i, (inputData, target) in enumerate(train_loader):
            inputData = inputData.cuda()
            target = target.cuda()  # (batch,3,num_classes)
            target = target.max(dim=1)[0]
            with autocast():  # mixed precision
                output = model(inputData).float()  # sigmoid will be done in loss !
            loss = criterion(output, target)
            model.zero_grad()

            scaler.scale(loss).backward()
            # loss.backward()

            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()

            scheduler.step()

            ema.update(model)
            # store information
            if i % 100 == 0:
                trainInfoList.append([epoch, i, loss.item()])
                print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                      .format(epoch, Epochs, str(i).zfill(3), str(steps_per_epoch).zfill(3),
                              scheduler.get_last_lr()[0], \
                              loss.item()))

        # try:
        #     torch.save(model.state_dict(), os.path.join(
        #         'models/', 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))
        # except:
        #     pass

        ## just normal evaluation, doesn't seem to be using any metalearning
        model.eval()

        for i_val, (inputData_val, target_val) in enumerate(val_loader):
            inputData_val = inputData_val.cuda()
            target_val = target_val.cuda()
            target_val = target_val.max(dim=1)[0]
            with autocast():
                output_val = model(inputData_val).float()
            loss_val = criterion(output_val, target_val)
            if i_val % 100 == 0:
                print('Val Loss valEpocw [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                      .format(epoch, Epochs, str(i_val).zfill(3), str(len(val_loader)).zfill(3),
                              scheduler.get_last_lr()[0], \
                              loss_val.item()))

        mAP_score = validate_multi(val_loader, model, ema)
        model.train()
        if mAP_score > highest_mAP:
            highest_mAP = mAP_score
            # try:
            #     torch.save(model.state_dict(), os.path.join(
            #         'models/', 'model-highest.ckpt'))
            # except:
            #     pass
        print('current_mAP = {:.2f}, highest_mAP = {:.2f}'.format(mAP_score, highest_mAP))

def validate_multi(val_loader, model, ema_model, print_info="yes", name = ''):
    print("starting validation")
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    preds_ema = []
    targets = []
    for i, (input, target) in enumerate(val_loader):
        target = target
        target = target.max(dim=1)[0]
        # compute output
        with torch.no_grad():
            with autocast():
                output_regular = Sig(model(input.cuda())).cpu()
                output_ema = Sig(ema_model.module(input.cuda())).cpu()

        # for mAP calculation
        preds_regular.append(output_regular.cpu().detach())
        preds_ema.append(output_ema.cpu().detach())
        targets.append(target.cpu().detach())

    mAP_score_regular = mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy(),print_info)
    mAP_score_ema = mAP(torch.cat(targets).numpy(), torch.cat(preds_ema).numpy(),print_info)
    print("mAP score regular {:.2f}, mAP score EMA {:.2f}".format(mAP_score_regular, mAP_score_ema))
    return max(mAP_score_regular, mAP_score_ema)

def getInverseClassFreqs():
    rootPath = '/raid/ganesh/prateekch/MSCOCO_2014'
    train_annotation_file_path = os.path.join(rootPath, 'annotations/instances_train2014.json')
    val_annotation_file_path = os.path.join(rootPath, 'annotations/instances_val2014.json')

    coco2014_train = coco2014(train_annotation_file_path)
    coco2014_val = coco2014(val_annotation_file_path)

    trainClassFreqs = coco2014_train.class_frequencies
    invClassFreqs = torch.tensor(10000/np.array(list(trainClassFreqs.values())), requires_grad=False, device='cuda')
    return invClassFreqs

## implementing without gradscaler as normal one isn't working, ot using lr scheduler as well, just plain simple training loop
## because the above one is not working 
class learner(nn.Module):
    def __init__(self, model_train, model_val, train_loader, val_loader, args):
        super(learner, self).__init__()
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
        self.stop_epoch = 30    ## can be updated during training process
        self.weight_decay = 1e-4

        self.steps_per_epoch_train = len(train_loader)
        self.steps_per_epoch_val = len(val_loader)

        self.optimizer_train = torch.optim.Adam(params=self.model_train.parameters(), lr=self.lr, weight_decay=0)  # optimizer for main model
        self.optimizer_val = torch.optim.Adam(params=self.model_val.parameters(), lr=self.meta_lr, weight_decay=0) # optimizer for meta model

        self.highest_mAP_train = 0
        self.highest_mAP_val = 0

        self.criteria = BCEloss()
        self.trainInfoList = []
        self.modeltrainvalinfo = []
        self.valInfoList = []

        if args.type == 'objpinv':
            self.weights = getInverseClassFreqs()
        else:
            self.weights = torch.rand(80).cuda() ## just initialising it randomly for the first round

        self.model_train.train()
        self.model_val.train()

    def forwardmax(self):
        print('FORWARD MAX-ING')
        for epoch in range(self.epochs):
            print(self.weights)
            if epoch > self.stop_epoch:
                break
            
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
                loss = self.criteria(output_train, target, self.weights)

                if i % 100 == 0 and self.args.print_info == "yes":
                    # printClassLoss(loss, "Max_Train")
                    print("Max Train Loss: ", loss)
                # print(loss.shape)
                loss = loss.max()   # this is max      ## max over classes

                optimizer_fasttrain.zero_grad()
                loss.backward(retain_graph=True)
                optimizer_fasttrain.step()

                ## ema model update
                # self.model_train_ema.update(fasttrain)

                if i % 100 == 0:
                    # self.trainInfoList.append([epoch, i, loss.item()])
                    print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
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
            for i_val, (inputData_val, target_val) in enumerate(self.val_loader):
                inputData_val = inputData_val.cuda()
                target_val = target_val.cuda()
                target_val = target_val.max(dim=1)[0]
                with autocast():
                    output_val = fasttrain(inputData_val).float()
                    # with torch.no_grad():
                    output_train_val = self.model_train(inputData_val).float()

                
                loss_val = self.criteria(output_val, target_val, self.weights)
                optimizer_fasttrain.zero_grad()

                if i_val % 100 == 0 and self.args.print_info == "yes":
                    # printClassLoss(loss_val, "Max_Val Meta Model")
                    print("Max_Val Meta Model: ", loss_val)

                loss_val = loss_val.max()
                loss_val.backward(retain_graph=True)
                optimizer_fasttrain.step()

                with torch.no_grad():
                    loss_train_val = self.criteria(output_train_val, target_val, self.weights)

                    if i_val % 100 == 0 and self.args.print_info == "yes":
                        # printClassLoss(loss_train_val, "Max_Val Main Model Weighted Val Loss")
                        print("Max_Val Main Model Weighted Val Loss: ", loss_train_val)

                    loss_train_val = loss_train_val.max()
                    loss_train_val_unweighted = self.criteria(output_train_val,target_val, torch.ones(self.weights.shape).cuda())
                    
                    if i_val % 100 == 0 and self.args.print_info == "yes":
                        # printClassLoss(loss_train_val_unweighted, "Max_Val Main Model Unweighted Val Loss")
                        print("Max_Val Main Model Unweighted Val Loss: ", loss_train_val_unweighted)

                    loss_train_val_unweighted = loss_train_val_unweighted.max()

                    gradients = []
                    for j, params in enumerate(fasttrain.parameters()):
                        gradients.append(deepcopy(params.grad))

                    for j, p in enumerate(self.model_val.parameters()):
                        p.grad = gradients[j]

                    self.optimizer_val.step()
                    self.optimizer_val.zero_grad()

                ## ema model update
                # self.model_val_ema.update(self.model_val)

                with autocast():
                    outputs_val = self.model_val(inputData_val).float()
                with torch.no_grad():
                    self.weights = torch.sigmoid(outputs_val.mean(dim=0))  ## updating weights

                if i_val % 100 == 0:
                    # self.valInfoList.append([epoch, i_val, loss_val.max().item()])
                    # self.modeltrainvalinfo.append([epoch, i_val, loss_train_val.item()])
                    print('Outer loop valEpocw Maximum [{}/{}], Step [{}/{}], LR {:.1e}, Meta Learning Max Validation Loss: {:.1f}'
                            .format(epoch, self.epochs, str(i_val).zfill(3), str(self.steps_per_epoch_val).zfill(3),
                                    self.optimizer_val.param_groups[0]['lr'], \
                                    loss_val.item()))
                    print('model_train val_loss valEpocw [{}/{}], Step [{}/{}], LR {:.1e}, Main Model Weighted Validation Loss: {:.1f}'
                            .format(epoch, self.epochs, str(i_val).zfill(3), str(self.steps_per_epoch_val).zfill(3),
                                    self.lr, \
                                    loss_train_val.item()))
                    print('model_train val_loss  valEpocw [{}/{}], Step [{}/{}], LR {:.1e}, Main Model Unweighted Validation Loss: {:.1f}'
                            .format(epoch, self.epochs, str(i_val).zfill(3), str(self.steps_per_epoch_val).zfill(3),
                                    self.lr, \
                                    loss_train_val_unweighted.item()))
                    # torch.save(self.model_val.state_dict(), os.path.join(
                    # 'models/{}/{}/models_val/'.format(self.type, self.model_name), 'model-{}-{}.ckpt'.format(epoch + 1, i_val + 1)))
                    # torch.save(self.weights, os.path.join(
                    # 'models/{}/{}/weights/'.format(self.type, self.model_name), 'weights-{}-{}.ckpt'.format(epoch + 1, i_val + 1)))  


            del fasttrain, parameters_fasttrain, optimizer_fasttrain
            torch.cuda.empty_cache()
            del gradients
            gc.collect()
            
            ## evaluation of the models
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
                #     'models/{}/{}/models_val/'.format(self.type, self.model_name), 'model-highest.ckpt'))    
            print('Val_data_mAP: current_mAP = {:.2f}, highest_mAP = {:.2f}'.format(mAP_score_val, self.highest_mAP_val))

    ## performing the training instead of max, taking sum at all places
    def forwardsum(self):
        print("FORWARD SUMMING")
        for epoch in range(self.epochs):
            print(f"Lambda Weights {self.weights}")
            if epoch > self.stop_epoch:
                break
            
            # meta model
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

                loss = loss*self.weights
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

            # Manoj - fasttrain is mainly for validation set checking of the main model.

            self.model_train.eval()
            fasttrain.eval()
            print(f"Lambda Weights {self.weights}")
            for i_val, (inputData_val, target_val) in enumerate(self.val_loader):
                inputData_val = inputData_val.cuda()
                target_val = target_val.cuda()
                target_val = target_val.max(dim=1)[0]

                with autocast():
                    output_val = fasttrain(inputData_val).float()
                    # output_train_val = self.model_train(inputData_val).float()
                loss_val = self.criteria(output_val, target_val) # validation loss
                loss_val = loss_val*self.weights # number_classes
                if i_val % 100 == 0 and self.args.print_info == "yes":
                    print("Sum_Val Meta Model: ", loss_val)
                
                if self.args.losscrit == 'sum':
                    loss_val = loss_val.sum()
                else:
                    loss_val = Gumbel(loss_val, tau=0.01, hard=False, eps=1e-10, dim=-1)
                
                optimizer_fasttrain.zero_grad()
                loss_val.backward(retain_graph=True)
                optimizer_fasttrain.step()
                
                # Manoj - Main Model on Validation Set
                
                with torch.no_grad():
                    with autocast():
                        output_train_val = self.model_train(inputData_val).float()  # main model output on validation set
                    loss_train_val_unweighted = self.criteria(output_train_val, target_val) # main model loss on validation set
                    loss_train_val = loss_train_val_unweighted*self.weights
                    
                    if i_val % 100 == 0 and self.args.print_info == "yes":
                        # printClassLoss(loss_train_val, "Sum_Val Main Model Weighted Val Loss")
                        print("Sum_Val Main Model Weighted Val Loss: ", loss_train_val)
                    loss_train_val = loss_train_val.sum()

                    if i_val % 100 == 0 and self.args.print_info == "yes":
                        # printClassLoss(loss_train_val_unweighted, "Sum_Val Main Model Unweighted Val Loss")
                        print("Sum_Val Main Model Unweighted Val Loss: ", loss_train_val_unweighted)
                    loss_train_val_unweighted = loss_train_val_unweighted.sum()
                    
                    gradients = []
                    for j, params in enumerate(fasttrain.parameters()):
                        gradients.append(deepcopy(params.grad))

                    for j, p in enumerate(self.model_val.parameters()):
                        p.grad = gradients[j]

                    self.optimizer_val.step()
                    self.optimizer_val.zero_grad()
                
                # with autocast():
                #     outputs_val = self.model_val(inputData_val).float() # main model output on the val set.
                # with torch.no_grad():
                #     self.weights = torch.sigmoid(outputs_val.mean(dim=0))
                
                if i_val % 100 == 0:
                    # self.valInfoList.append([epoch, i_val, loss_val.max().item()])
                    # self.modeltrainvalinfo.append([epoch, i_val, loss_train_val.item()])
                    print('Outer loop valEpocw Maximum [{}/{}], Step [{}/{}], LR {:.1e}, Meta Learning Summed up Validation Loss: {:.1f}'
                          .format(epoch, self.epochs, str(i_val).zfill(3), str(self.steps_per_epoch_val).zfill(3),
                                  self.optimizer_val.param_groups[0]['lr'], \
                                  loss_val.sum().item()))
                    print('model_train val_loss valEpocw [{}/{}], Step [{}/{}], LR {:.1e}, Main Model Validation Loss: {:.1f}'
                            .format(epoch, self.epochs, str(i_val).zfill(3), str(self.steps_per_epoch_val).zfill(3),
                                    self.lr, \
                                    loss_train_val.item()))
                    
                    print('model_train val_loss valEpocw [{}/{}], Step [{}/{}], LR {:.1e}, Main Model Unweighted Validation Loss: {:.1f}'
                            .format(epoch, self.epochs, str(i_val).zfill(3), str(self.steps_per_epoch_val).zfill(3),
                                    self.lr, \
                                    loss_train_val_unweighted.item()))
                    # torch.save(self.model_val.state_dict(), os.path.join(
                    # 'models/{}/{}/models_val/'.format(self.type, self.model_name), 'model-{}-{}.ckpt'.format(epoch + 1, i_val + 1)))
                    # torch.save(self.weights, os.path.join(
                    # 'models/{}/{}/weights/'.format(self.type, self.model_name), 'weights-{}-{}.ckpt'.format(epoch + 1, i_val + 1)))
                
            # self.model_train.train()
            del fasttrain, parameters_fasttrain, optimizer_fasttrain
            torch.cuda.empty_cache()
            del gradients
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