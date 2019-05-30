import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import numpy as np
import pickle
import cv2
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import sys
import os
from tqdm import tqdm
import os.path as osp
from networks.deeplab import DeepLabv3
from dataset.datasets import *
import random
import timeit
import logging
from tensorboardX import SummaryWriter
from utils.utils import decode_labels, inv_preprocess, decode_predictions
from collections import OrderedDict
import re

start = timeit.default_timer()

IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)

BATCH_SIZE = 16
DATA_DIRECTORY = 'cityscapes'
DATA_LIST_PATH = './dataset/list/voc/train.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '513,513'
LEARNING_RATE = 0.007
MOMENTUM = 0.9
NUM_CLASSES = 21
NUM_STEPS = 30000
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = './dataset/resnet101-imagenet.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 5000
SNAPSHOT_DIR = 'snapshots/'
WEIGHT_DECAY = 0.0001

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--start-iters", type=int, default=0,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=str, default='None',
                        help="choose gpu device.")
    parser.add_argument("--ft", type=bool, default=False,
                        help="fine-tune the model with large input size.")
    parser.add_argument("--aug", type=bool, default=False,
                        help="")

    return parser.parse_args()

args = get_arguments()


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))
            
def adjust_learning_rate(optimizer, i_iter):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr
    return lr

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def set_bn_momentum(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1 or classname.find('InPlaceABN') != -1:
        m.momentum = 0.0003

def load_resnet(state_dict, num_layers):
    def reindex_reformat_layer_block(name):
        name = name.group(0)
        return 'layer' + str(int(name[5])+1) + '.block' + str(int(name[7])+1) + '.'
    def reindex_reformat_layer_block_dd(name):
        name = name.group(0)
        return 'layer' + str(int(name[5])+1) + '.block' + str(int(name[7:])+1)
    state_dict = OrderedDict([(re.sub('layer[1-4]\.[0-9]\.', reindex_reformat_layer_block, k), v) for k, v in state_dict.items()])
    state_dict = OrderedDict([(re.sub('layer[1-4]\.[0-9][0-9]', reindex_reformat_layer_block_dd, k), v) for k, v in state_dict.items()])
    state_dict = OrderedDict([(k.replace('downsample.0', 'shortcut.conv'), v) for k, v in state_dict.items()])
    state_dict = OrderedDict([(k.replace('downsample.1', 'shortcut.bn'), v) for k, v in state_dict.items()])
    state_dict = OrderedDict([(k.replace('conv1', 'reduce'), v) for k, v in state_dict.items()])
    state_dict = OrderedDict([(k.replace('conv3', 'increase'), v) for k, v in state_dict.items()])
    state_dict = OrderedDict([(k.replace('conv2', 'conv3x3'), v) for k, v in state_dict.items()])
    state_dict = OrderedDict([(k.replace('bn1', 'reduce.bn'), v) for k, v in state_dict.items()])
    state_dict = OrderedDict([(k.replace('bn2', 'conv3x3.bn'), v) for k, v in state_dict.items()])
    state_dict = OrderedDict([(k.replace('bn3', 'increase.bn'), v) for k, v in state_dict.items()])
    state_dict = OrderedDict([(k.replace('weight', 'conv.weight'), v) if 'bn' not in k else (k,v) for k, v in state_dict.items()])
    state_dict = OrderedDict([(k.replace('reduce', 'layer1.conv1'), v) if k.startswith('reduce') else (k,v) for k, v in state_dict.items()])
    state_dict = OrderedDict([(k.replace('conv.conv', 'conv'), v) for k, v in state_dict.items()])
    return state_dict

def main():
    """Create the model and start the training."""
    writer = SummaryWriter(args.snapshot_dir)
    
    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    cudnn.enabled = True

    # Create network.

    model = DeepLabv3(num_classes=args.num_classes, output_stride=8)
    state_dict = torch.load(args.restore_from)
    '''if 'resnet50' in args.restore_from:
        state_dict = load_resnet(state_dict, 50)
    elif 'resnet101' in args.restore_from:
        state_dict = load_resnet(state_dict, 101)'''
    state_dict = OrderedDict([(k,v) for k,v in state_dict.items() if v is not None])
    state_dict = OrderedDict([(k.replace('module.', ''), v) for k, v in state_dict.items()])
    model.load_state_dict(state_dict, strict=False)  # to skip ASPP


    model = nn.DataParallel(model)
    model.train()
    model.float()
    #model.apply(set_bn_momentum)
    model.cuda()

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    criterion.cuda()
    
    cudnn.benchmark = True

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    if args.aug:
        args.data_list = './dataset/list/voc/train_aug.txt'

    trainloader = data.DataLoader(VOCDataSet(args.data_dir, args.data_list, max_iters=args.num_steps*args.batch_size, crop_size=input_size, 
                    scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN, aug=args.aug), 
                    batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    optimizer = optim.SGD([{'params': filter(lambda p: p.requires_grad, [p for n, p in model.named_parameters() if 'aspp' in n or 'classifier' in n]), 'lr': args.learning_rate},
                          {'params': filter(lambda p: p.requires_grad, [p for n,p in model.named_parameters() if 'aspp' not in n and 'classifier' not in n]), 'lr': args.learning_rate}], 
                lr=args.learning_rate, momentum=args.momentum,weight_decay=args.weight_decay)
    optimizer.zero_grad()

    trainloader = tqdm(trainloader)

    for i_iter, batch in enumerate(trainloader):
        i_iter += args.start_iters
        images, labels, _, _ = batch
        images = images.cuda()
        labels = labels.long().cuda()

        optimizer.zero_grad()
        lr = adjust_learning_rate(optimizer, i_iter)
        preds = model(images)
        preds = F.interpolate(preds, size=labels.shape[2:])

        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        if i_iter % 100 == 0:
            writer.add_scalar('learning_rate', lr, i_iter)
            writer.add_scalar('loss', loss.data.cpu().numpy(), i_iter)

        # if i_iter % 5000 == 0:
        #     images_inv = inv_preprocess(images, args.save_num_images, IMG_MEAN)
        #     labels_colors = decode_labels(labels, args.save_num_images, args.num_classes)
        #     if isinstance(preds, list):
        #         preds = preds[0]
        #     preds_colors = decode_predictions(preds, args.save_num_images, args.num_classes)
        #     for index, (img, lab) in enumerate(zip(images_inv, labels_colors)):
        #         writer.add_image('Images/'+str(index), img, i_iter)
        #         writer.add_image('Labels/'+str(index), lab, i_iter)
        #         writer.add_image('preds/'+str(index), preds_colors[index], i_iter)

        #print('iter = {} of {} completed, loss = {}'.format(i_iter, args.num_steps, loss.data.cpu().numpy()))

        if i_iter >= args.num_steps-1:
            torch.save(model.state_dict(),osp.join(args.snapshot_dir, 'CS_scenes_'+str(args.num_steps)+'.pth'))
            break

        if i_iter % args.save_pred_every == 0:
            torch.save(model.state_dict(),osp.join(args.snapshot_dir, 'CS_scenes_'+str(i_iter)+'.pth'))     

    end = timeit.default_timer()
    print(end-start,'seconds')

if __name__ == '__main__':
    main()
