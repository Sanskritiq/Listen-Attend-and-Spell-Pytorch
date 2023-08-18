# import yaml
# import os
# from util.librispeech_dataset import create_dataloader
# from util.functions import log_parser,batch_iterator, collapse_phn
# from model.las_model import Listener,Speller
# import numpy as np
# from torch.autograd import Variable
# import torch
# import time
# from tensorboardX import SummaryWriter

# import argparse


# parser = argparse.ArgumentParser(description='Training script for LAS on Librispeech .')

# parser.add_argument('config_path', metavar='config_path', type=str,
#                      help='Path to config file for training.')

# paras = parser.parse_args()

# config_path = paras.config_path

# # Load config file for experiment
# print('Loading configure file at',config_path)
# conf = yaml.load(open(config_path,'r'), Loader=yaml.Loader)

# # Parameters loading
# print()
# print('Experiment :',conf['meta_variable']['experiment_name'])
# total_steps = conf['training_parameter']['total_steps']

# listener_model_path = conf['meta_variable']['checkpoint_dir']+conf['meta_variable']['experiment_name']+'.listener'
# speller_model_path = conf['meta_variable']['checkpoint_dir']+conf['meta_variable']['experiment_name']+'.speller'
# verbose_step = conf['training_parameter']['verbose_step']
# valid_step = conf['training_parameter']['valid_step']
# tf_rate_upperbound = conf['training_parameter']['tf_rate_upperbound']
# tf_rate_lowerbound = conf['training_parameter']['tf_rate_lowerbound']
# tf_decay_step = conf['training_parameter']['tf_decay_step']
# seed = conf['training_parameter']['seed']

# # Fix random seed
# np.random.seed(seed)
# torch.manual_seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(seed)

# # Load preprocessed LibriSpeech Dataset

# train_set = create_dataloader(conf['meta_variable']['data_path']+'/train.csv', 
#                               **conf['model_parameter'], **conf['training_parameter'], shuffle=True,training=True)
# valid_set = create_dataloader(conf['meta_variable']['data_path']+'/dev.csv',
#                               **conf['model_parameter'], **conf['training_parameter'], shuffle=False,drop_last=True)

# idx2char = {}
# with open(conf['meta_variable']['data_path']+'/idx2chap.csv','r') as f:
#     for line in f:
#         if 'idx' in line:continue
#         idx2char[int(line.split(',')[0])] = line[:-1].split(',')[1]

# # Load pre-trained model if needed
# if conf['training_parameter']['use_pretrained']:
#     # global_step = conf['training_parameter']['pretrained_step']
#     listener = torch.load(conf['training_parameter']['pretrained_listener_path'], map_location=torch.device('cpu'))
#     speller = torch.load(conf['training_parameter']['pretrained_speller_path'], map_location=torch.device('cpu'))
# else:
#     # global_step = 0
#     listener = Listener(**conf['model_parameter'])
#     speller = Speller(**conf['model_parameter'])
# optimizer = torch.optim.Adam([{'params':listener.parameters()}, {'params':speller.parameters()}], 
#                               lr=conf['training_parameter']['learning_rate'])

# best_ler = 1.0
# record_gt_text = False
# log_writer = SummaryWriter(conf['meta_variable']['training_log_dir']+conf['meta_variable']['experiment_name'])

import yaml
from util.timit_dataset import load_dataset,create_dataloader
from model.las_model import Listener,Speller
from util.functions import batch_iterator
import numpy as np
from torch.autograd import Variable
import torch
import sys
from tensorboardX import SummaryWriter
import argparse

# Load config file for experiment
parser = argparse.ArgumentParser(description='Training script for LAS on TIMIT .')

parser.add_argument('config_path', metavar='config_path', type=str,
                     help='Path to config file for training.')

paras = parser.parse_args()

config_path = paras.config_path

conf = yaml.safe_load(open(config_path,'r'))


# Parameters loading
torch.manual_seed(conf['training_parameter']['seed'])
total_steps = conf['training_parameter']['total_steps']
use_pretrained = conf['training_parameter']['use_pretrained']
verbose_step = conf['training_parameter']['verbose_step']
valid_step  = conf['training_parameter']['valid_step']
tf_rate_upperbound = conf['training_parameter']['tf_rate_upperbound']
tf_rate_lowerbound = conf['training_parameter']['tf_rate_lowerbound']

# # Load preprocessed TIMIT Dataset ( using testing set directly here, replace them with validation set your self)
# # X : Padding to shape [num of sample, max_timestep, feature_dim]
# # Y : Squeeze repeated label and apply one-hot encoding (preserve 0 for <sos> and 1 for <eos>)
# X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(**conf['meta_variable'])
# train_set = create_dataloader(X_train, y_train, **conf['model_parameter'], **conf['training_parameter'], shuffle=True)
# valid_set = create_dataloader(X_val, y_val, **conf['model_parameter'], **conf['training_parameter'], shuffle=False)
# #test_set = create_dataloader(X_test, y_test, **conf['model_parameter'], **conf['training_parameter'], shuffle=False)

# Construct LAS Model or load pretrained LAS model
log_writer = SummaryWriter(conf['meta_variable']['training_log_dir']+conf['meta_variable']['experiment_name'])

if not use_pretrained:
    listener = Listener(**conf['model_parameter'])
    speller = Speller(**conf['model_parameter'])
else:
    listener = torch.load(conf['training_parameter']['pretrained_listener_path'], map_location=torch.device('cpu'))
    speller = torch.load(conf['training_parameter']['pretrained_speller_path'], map_location=torch.device('cpu'))
optimizer = torch.optim.Adam([{'params':listener.parameters()}, {'params':speller.parameters()}],
                             lr=conf['training_parameter']['learning_rate'])
listener_model_path = conf['meta_variable']['checkpoint_dir']+conf['meta_variable']['experiment_name']+'.listener'
speller_model_path = conf['meta_variable']['checkpoint_dir']+conf['meta_variable']['experiment_name']+'.speller'

# save checkpoint with the best ler
best_ler = 1.0
global_step = 0

