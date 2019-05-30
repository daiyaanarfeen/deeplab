import argparse
import scipy
from scipy import ndimage
import cv2
import numpy as np
import sys
import json

import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data
from torch.nn.parallel.distributed import DistributedDataParallel
from networks.deeplab import DeepLabv3
from dataset.datasets import VOCDataSet
from collections import OrderedDict
import os
import scipy.ndimage as nd
from math import ceil
from PIL import Image as PILImage
from utils.ptwarp import warp_using_logits, inverse_warp_using_logits

import torch.nn as nn
IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)

DATA_DIRECTORY = 'cityscapes'
DATA_LIST_PATH = './dataset/list/voc/val.txt'
IGNORE_LABEL = 255
NUM_CLASSES = 21
NUM_STEPS = 1449 # Number of images in the validation set.
INPUT_SIZE = '513,513'
RESTORE_FROM = './deeplab_resnet.ckpt'

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=str, default='0',
                        help="choose gpu device.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--whole", type=bool, default=False,
                        help="use whole input size.")
    parser.add_argument("--warp", type=bool, default=False, 
                        help="warp")
    return parser.parse_args()

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """

    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def pad_image(img, target_size):
    """Pad an image up to the target size."""
    rows_missing = target_size[0] - img.shape[2]
    cols_missing = target_size[1] - img.shape[3]
    padded_img = np.pad(img, ((0, 0), (0, 0), (0, rows_missing), (0, cols_missing)), 'constant')
    return padded_img

def predict_whole(net, image, tile_size):
    interp = nn.Upsample(size=tile_size, mode='bilinear', align_corners=True)
    prediction = net(image)
    if isinstance(prediction, list):
        prediction = prediction[0]
    prediction = prediction.data.permute(0,2,3,1)
    return prediction

def predict_multiscale(net, image, tile_size, scales, classes, flip_evaluation):
    """
    Predict an image by looking at it with different scales.
        We choose the "predict_whole_img" for the image with less than the original input size,
        for the input of larger size, we would choose the cropping method to ensure that GPU memory is enough.
    """
    image = image.data
    N_, C_, H_, W_ = image.shape
    full_probs = torch.zeros((N_, H_, W_, classes))
    for scale in scales:
        scale = float(scale)
        #print("Predicting image scaled by %f" % scale)
        scale_image = F.interpolate(image, scale_factor=(scale, scale), mode='bilinear', align_corners=True)
        scaled_probs = predict_whole(net, scale_image, tile_size).cpu()
        if flip_evaluation == True:
            flip_scaled_probs = predict_whole(net, scale_image.flip(3), tile_size).cpu()
            scaled_probs = (0.5 * (scaled_probs + flip_scaled_probs.flip(2)))
            scaled_probs = F.interpolate(scaled_probs.permute(0, 3, 1, 2), size=(image.shape[2], image.shape[3]), mode='bilinear').permute(0, 2, 3, 1)
        full_probs += scaled_probs
    full_probs /= len(scales)
    return full_probs

def predict_warp(net, image, tile_size, classes):
    image = image.data
    N_, C_, H_, W_ = image.shape
    full_probs = torch.zeros((N_, H_, W_, classes))
    reg_logits = predict_whole(net, image, tile_size, 0).cpu()
    warped_image = warp_using_logits(image.permute(0, 2, 3, 1), reg_logits, max_distortion=2.0, min_distortion=0.1, num_classes=19)
    warp_logits = inverse_warp_using_logits(predict_whole(net, warped_image, tile_size, 0).cpu(), reg_logits, max_distortion=2.0, min_distortion=0.1, num_classes=19).permute(0, 2, 3, 1)
    full_probs = 0.5 * (reg_logits.cpu() + warp_logits.cpu())
    return full_probs

def get_confusion_matrix(gt_label, pred_label, class_num):
        """
        Calcute the confusion matrix by given label and pred
        :param gt_label: the ground truth label
        :param pred_label: the pred label
        :param class_num: the nunber of class
        :return: the confusion matrix
        """
        index = (gt_label * class_num + pred_label).astype('int32')
        label_count = np.bincount(index)
        confusion_matrix = np.zeros((class_num, class_num))

        for i_label in range(class_num):
            for i_pred_label in range(class_num):
                cur_index = i_label * class_num + i_pred_label
                if cur_index < len(label_count):
                    confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

        return confusion_matrix

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()

    # gpu0 = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    h, w = map(int, args.input_size.split(','))
    if args.whole or args.warp:
        input_size = (513, 513)
    else:
        input_size = (h, w)

    model = DeepLabv3(num_classes=args.num_classes, output_stride=8)
    
    saved_state_dict = torch.load(args.restore_from)
    saved_state_dict = OrderedDict([(k,v) for k,v in saved_state_dict.items() if v is not None])
    saved_state_dict = OrderedDict([(k.replace('module.', ''), v) for k, v in saved_state_dict.items()])
    model.load_state_dict(saved_state_dict)
    model = nn.DataParallel(model)

    model.eval()
    model.cuda()

    testloader = data.DataLoader(VOCDataSet(args.data_dir, args.data_list, crop_size=(513, 513), mean=IMG_MEAN, scale=False, mirror=False), 
                                    batch_size=32, shuffle=False, pin_memory=True)

    data_list = []
    confusion_matrix = np.zeros((args.num_classes,args.num_classes))
    palette = get_palette(256)
    interp = nn.Upsample(size=(513, 513), mode='bilinear', align_corners=True)

    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    for index, batch in enumerate(testloader):
        print('%d processd out of %d'%(index, len(testloader)) )
        image, label, size, name = batch
        size = size[0].numpy()
        with torch.no_grad():
            if args.whole:
                output = predict_multiscale(model, image, input_size, [0.75, 1.0, 1.25], args.num_classes, True)
            elif args.warp:
                output = predict_warp(model, image, input_size, args.num_classes)
            else:
                output = predict_sliding(model, image.numpy(), input_size, args.num_classes, True)
        # padded_prediction = model(Variable(image, volatile=True).cuda())
        # output = interp(padded_prediction).cpu().data[0].numpy().transpose(1,2,0)
        seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
        #output_im = PILImage.fromarray(seg_pred)
        #output_im.putpalette(palette)
        #output_im.save('outputs/'+name[0]+'.png')

        seg_gt = np.asarray(label.numpy(), dtype=np.int)
        #ignore_index = seg_gt != 255
        #seg_gt = seg_gt[ignore_index]
        #seg_pred = seg_pred[ignore_index]

        for i in range(seg_pred.shape[0]):
            pred = seg_pred[i]
            gt = seg_gt[i]     
            ignore_index = gt != 255
            gt = gt[ignore_index]
            pred = pred[ignore_index]
            # show_all(gt, output)
            confusion_matrix += get_confusion_matrix(gt, pred, args.num_classes)

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)

    IU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IU = IU_array.mean()
    
    # getConfusionMatrixPlot(confusion_matrix)
    print({'meanIU':mean_IU, 'IU_array':IU_array})
    with open('result.txt', 'w') as f:
        f.write(json.dumps({'meanIU':mean_IU, 'IU_array':IU_array.tolist()}))

if __name__ == '__main__':
    main()
