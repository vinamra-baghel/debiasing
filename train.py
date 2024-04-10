import os
import argparse
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
from src.helper_functions.helper_functions import CocoDetection, CutoutPIL
from src.models import create_model
from randaugment import RandAugment

from cocostats import coco2014
from learner import learner, train_multi_label_coco
from bilevelopt import BiLevelOpt

def parseArgs():
    parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')
    parser.add_argument('--data', metavar='DIR', help='path to dataset', default='/raid/ganesh/prateekch/MSCOCO_2014')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--meta-lr', default=1e-4, type=float)
    parser.add_argument('--model-name', default='tresnet_m')
    parser.add_argument('--model-path', default='./models_local/MS_COCO_TRresNet_M_224_81.8.pth', type=str)
    parser.add_argument('--num-classes', default=80)
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--image-size', default=224, type=int,
                        metavar='N', help='input image size (default: 448)')
    parser.add_argument('--thre', default=0.8, type=float,
                        metavar='N', help='threshold value')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--print-freq', '-p', default=64, type=int,
                        metavar='N', help='print frequency (default: 64)')
    parser.add_argument('--type','-t', default='bilevel', type=str)     ## added this argument to differentiate between various methods
    parser.add_argument('--losscrit', type=str, default='sum')
    parser.add_argument('--reg', default=0)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--print-info', default='no', type=str)
    return parser.parse_args()

def main():
    args = parseArgs()
    ## random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.do_bottleneck_head = False

    if args.type == 'asl':
        print('creating model for ASL...')
        model = create_model(args).cuda()

        if args.model_path:  # make sure to load pretrained ImageNet model
            state = torch.load(args.model_path, map_location='cpu')
            filtered_dict = {k: v for k, v in state['model'].items() if
                         (k in model.state_dict() and 'head.fc' not in k)}
            model.load_state_dict(filtered_dict, strict=False)
        print('done\n')
    
    elif 'objp' in args.type or args.type == 'bce' or args.type == 'bilevel':
        print('creating models for Debiasing Objective...')
        
        # added two models for train and val purpose
        model_train = create_model(args).cuda()
        model_val = create_model(args).cuda()

        if args.model_path:  # make sure to load pretrained ImageNet model
            state = torch.load(args.model_path, map_location='cpu')
            filtered_dict = {k: v for k, v in state['model'].items() if
                         (k in model_train.state_dict() and 'head.fc' not in k)}
            model_train.load_state_dict(filtered_dict, strict=False)

            filtered_dict = {k: v for k, v in state['model'].items() if
                            (k in model_val.state_dict() and 'head.fc' not in k)}
            model_val.load_state_dict(filtered_dict, strict=False)

        print('done\n')

    # COCO Data loading
    instances_path_val = os.path.join(args.data, 'annotations/instances_val2014.json')
    instances_path_train = os.path.join(args.data, 'annotations/instances_train2014.json')
    data_path_val   = f'{args.data}/val2014'    # args.data
    data_path_train = f'{args.data}/train2014'  # args.data
    val_dataset = CocoDetection(data_path_val,
                                instances_path_val,
                                transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    # normalize, # no need, toTensor does normalization
                                ]))
    train_dataset = CocoDetection(data_path_train,
                                  instances_path_train,
                                  transforms.Compose([
                                      transforms.Resize((args.image_size, args.image_size)),
                                      CutoutPIL(cutout_factor=0.5),
                                      RandAugment(),
                                      transforms.ToTensor(),
                                      # normalize,
                                  ]))
    print("len(val_dataset)): ", len(val_dataset))
    print("len(train_dataset)): ", len(train_dataset))

    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    learnerObj = learner(model_train, model_val, train_loader, val_loader, args)
    bilevelObj = BiLevelOpt(model_train, model_val, train_loader, val_loader, args)

    # Actual Training
    if args.type == 'asl':
        train_multi_label_coco(model, train_loader, val_loader, args.lr)
    elif args.type == 'bilevel':
        bilevelObj()
    elif args.type == 'objpmax':
        learnerObj.forwardmax()
    elif args.type == 'objpsum' or args.type == 'objpinv':
        learnerObj.forwardsum()

if __name__ == '__main__':
    main()