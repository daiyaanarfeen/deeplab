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
from networks.deeplab import DeepLabv3
from dataset.datasets import CSDataSet
from collections import OrderedDict
import os
import scipy.ndimage as nd
from math import ceil
from PIL import Image as PILImage
from utils.ptwarp import warp_using_logits, inverse_warp_using_logits

import torch.nn as nn
IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)

DATA_DIRECTORY = 'cityscapes'
DATA_LIST_PATH = './dataset/list/cityscapes/val.lst'
IGNORE_LABEL = 255
NUM_CLASSES = 19
NUM_STEPS = 500 # Number of images in the validation set.
INPUT_SIZE = '769,769'
RESTORE_FROM = './deeplab_resnet.ckpt'

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="")
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
    parser.add_argument("--recurrence", type=int, default=1,
                        help="choose the number of recurrence.")
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

def predict_sliding(net, image, tile_size, classes, flip_evaluation, recurrence):
    interp = nn.Upsample(size=tile_size, mode='bilinear', align_corners=True)
    image_size = image.shape
    overlap = 1.0/3.0

    stride = ceil(tile_size[0] * (1 - overlap))
    tile_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)  # strided convolution formula
    tile_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)
    print("Need %i x %i prediction tiles @ stride %i px" % (tile_cols, tile_rows, stride))
    full_probs = np.zeros((image_size[2], image_size[3], classes))
    count_predictions = np.zeros((image_size[2], image_size[3], classes))
    tile_counter = 0

    for row in range(tile_rows):
        for col in range(tile_cols):
            x1 = int(col * stride)
            y1 = int(row * stride)
            x2 = min(x1 + tile_size[1], image_size[3])
            y2 = min(y1 + tile_size[0], image_size[2])
            x1 = max(int(x2 - tile_size[1]), 0)  # for portrait images the x1 underflows sometimes
            y1 = max(int(y2 - tile_size[0]), 0)  # for very few rows y1 underflows

            img = image[:, :, y1:y2, x1:x2]
            padded_img = pad_image(img, tile_size)
            # plt.imshow(padded_img)
            # plt.show()
            tile_counter += 1
            print("Predicting tile %i" % tile_counter)
            padded_prediction = net(Variable(torch.from_numpy(padded_img), volatile=True).cuda(), recurrence)
            if isinstance(padded_prediction, list):
                padded_prediction = padded_prediction[0]
            padded_prediction = interp(padded_prediction).cpu().data[0].numpy().transpose(1,2,0)
            prediction = padded_prediction[0:img.shape[2], 0:img.shape[3], :]
            count_predictions[y1:y2, x1:x2] += 1
            full_probs[y1:y2, x1:x2] += prediction  # accumulate the predictions also in the overlapping regions

    # average the predictions in the overlapping regions
    full_probs /= count_predictions
    # visualize normalization Weights
    # plt.imshow(np.mean(count_predictions, axis=2))
    # plt.show()
    return full_probs

def predict_whole(net, image, tile_size, recurrence):
    interp = nn.Upsample(size=tile_size, mode='bilinear', align_corners=True)
    prediction = net(image)
    if isinstance(prediction, list):
        prediction = prediction[0]
    prediction = interp(prediction).data.permute(0,2,3,1)
    return prediction

def predict_multiscale(net, image, tile_size, scales, classes, flip_evaluation, recurrence):
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
        scaled_probs = predict_whole(net, scale_image, tile_size, recurrence).cpu()
        if flip_evaluation == True:
            flip_scaled_probs = predict_whole(net, scale_image.flip(3), tile_size, recurrence).cpu()
            scaled_probs = 0.5 * (scaled_probs + flip_scaled_probs.flip(2))
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
        input_size = (1024, 2048)
    else:
        input_size = (h, w)

    model = DeepLabv3(num_classes=args.num_classes, output_stride=8)
    
    saved_state_dict = torch.load(args.restore_from)
    saved_state_dict = OrderedDict([(k.replace('module.', ''), v) for k, v in saved_state_dict.items()])
    model.load_state_dict(saved_state_dict)
    model = nn.DataParallel(model)

    model.eval()
    model.cuda()

    testloader = data.DataLoader(CSDataSet(args.data_dir, args.data_list, crop_size=(1024, 2048), mean=IMG_MEAN, scale=False, mirror=False), 
                                    batch_size=8, shuffle=False, pin_memory=True)

    data_list = []
    confusion_matrix = np.zeros((args.num_classes,args.num_classes))
    palette = get_palette(256)
    interp = nn.Upsample(size=(1024, 2048), mode='bilinear', align_corners=True)

    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    for index, batch in enumerate(testloader):
        print('%d processd out of %d'%(index, len(testloader)) )
        image, label, size, name = batch
        size = size[0].numpy()
        with torch.no_grad():
            if args.whole:
                output = predict_multiscale(model, image, input_size, [0.75, 1.0, 1.25], args.num_classes, True, args.recurrence)
            elif args.warp:
                output = predict_warp(model, image, input_size, args.num_classes)
            else:
                output = predict_sliding(model, image.numpy(), input_size, args.num_classes, True, args.recurrence)
        # padded_prediction = model(Variable(image, volatile=True).cuda())
        # output = interp(padded_prediction).cpu().data[0].numpy().transpose(1,2,0)
        seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
        #output_im = PILImage.fromarray(seg_pred)
        #output_im.putpalette(palette)
        #output_im.save('outputs/'+name[0]+'.png')

        seg_gt = np.asarray(label.numpy()[:size[0],:size[1]], dtype=np.int)

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
