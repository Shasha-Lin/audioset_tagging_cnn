import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import h5py
import math
import time
import logging
import matplotlib.pyplot as plt

import torch
# torch.backends.cudnn.benchmark=True
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from pytorch.models import *
from utils.utilities import get_filename
import librosa
import soundfile
from birds import config
# from .models import (Cnn14, Cnn14_no_specaug, Cnn14_no_dropout,
#     Cnn6, Cnn10, ResNet22, ResNet38, ResNet54, Cnn14_emb512, Cnn14_emb128,
#     Cnn14_emb32, MobileNetV1, MobileNetV2, LeeNet11, LeeNet24, DaiNet19,
#     Res1dNet31, Res1dNet51, Wavegram_Cnn14, Wavegram_Logmel_Cnn14,
#     Wavegram_Logmel128_Cnn14, Cnn14_16k, Cnn14_8k, Cnn14_mel32, Cnn14_mel128,
#     Cnn14_mixup_time_domain, Cnn14_DecisionLevelMax, Cnn14_DecisionLevelAtt)

pretrained_class_num = 527


class Transfer_Cnn14(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, fine_tune_classes_num, freeze_base=True, model_type='Wavegram_Logmel_Cnn14'):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(Transfer_Cnn14, self).__init__()
        base = eval(model_type)
        self.base = base(sample_rate, window_size, hop_size, mel_bins, fmin,
            fmax, pretrained_class_num)

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(2048, fine_tune_classes_num, bias=True)

        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, check_point_model):
        self.base.load_state_dict(check_point_model)

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)
        """
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict['embedding']

        clipwise_output = self.fc_transfer(embedding)
        clipwise_output = torch.sigmoid(clipwise_output)
        output_dict['clipwise_output'] = clipwise_output

        # labels = torch.zeros(clipwise_output.shape).cpu()
        # labels[torch.arange(clipwise_output.shape[0]), clipwise_output.argmax(dim=1)] = 1
        # labels = torch.argmax(clipwise_output, dim=1)
        # output_dict['labels'] = labels
        return output_dict


def train(args):

    # Arugments & parameters
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    pretrained_checkpoint_path = args.pretrained_checkpoint_path
    freeze_base = args.freeze_base
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'

    sample_rate = config.sample_rate
    classes_num = config.classes_num
    pretrain = True if pretrained_checkpoint_path else False
    
    # Model
    Model = eval(model_type)
    model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, 
        classes_num, freeze_base)

    # Load pretrained model
    if pretrain:
        logging.info('Load pretrained model from {}'.format(pretrained_checkpoint_path))
        model.load_from_pretrain(pretrained_checkpoint_path)

    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if 'cuda' in device:
        model.to(device)

    print('Load pretrained model successfully!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    # Train
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--window_size', type=int, required=True)
    parser_train.add_argument('--hop_size', type=int, required=True)
    parser_train.add_argument('--mel_bins', type=int, required=True)
    parser_train.add_argument('--fmin', type=int, required=True)
    parser_train.add_argument('--fmax', type=int, required=True) 
    parser_train.add_argument('--model_type', type=str, required=True)
    parser_train.add_argument('--pretrained_checkpoint_path', type=str)
    parser_train.add_argument('--freeze_base', action='store_true', default=False)
    parser_train.add_argument('--cuda', action='store_true', default=False)

    # Parse arguments
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        train(args)

    else:
        raise Exception('Error argument!')