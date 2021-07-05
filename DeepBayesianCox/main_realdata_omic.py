import pandas as pd
import numpy as np
import csv

from DeepBayesian import CREN_create
from DeepBayesian import CREN

import torch
from support import get_omic_data
from support import split_data
from support import sort_data
from support import mkdir
from support import plot_curve
from support import cal_pval
from DeepBayesian.utils import concordance_index
#---------------------------------------------------------------------------------



##nn_config : is_random_bias True means using DBP methods False means using CoxNN



#---------------------------------------------------------------------------------

# print(torch.cuda.current_device())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
is_plot = False

nn_config = {
    "learning_rate": 0.0000007,
    "learning_rate_decay": 0.999,
    "activation": 'selu',
    "epoch_num": 500,
    "skip_num": 5,
    "is_random_bias":True,
    "random_rate":1.5,
    "censor_restore_rate":0.2,
    "L1_reg": 3.4e-5,
    "L2_reg": 1.2e-4,
    "optimizer": 'Adam',
    "dropout": 0.0,
    "hidden_layers": [500, 200, 24, 1],
    "standardize":True,
    "batchnorm":False,
    "momentum": 0.7,
    "alpha": 0,
    "seed": 123,
    "torch_seed": 123
}

dataset = ["my_dataset/brca_data/BRCA"]
for filename in dataset:
    average_Cindex = 0
    test_num = 10
    dataX, dataY, headers= get_omic_data(fea_filename=(filename + ".csv"))
    for seed in range(test_num):
        for fold_num in range(1):
            ori_train_X, ori_train_Y, ori_test_X, ori_test_Y, ori_train_H, ori_test_H = split_data(spilt_seed=seed, fea=dataX, label=dataY, headers=headers, nfold=5, fold_num=fold_num)
            ori_idx, ori_train_X, ori_train_Y = sort_data(ori_train_X, ori_train_Y)
            input_nodes = len(ori_train_X[0])
            net = CREN_create(nn_config, input_nodes).to(device)
            model = CREN(nn_config, net)
            model._train(device, ori_train_X, ori_train_Y)
            prediction = model._predict(device, ori_test_X)
            Cindex = concordance_index(ori_test_Y, -prediction)
            print("Cindex is : " + str(Cindex))
            average_Cindex += Cindex
    print("Average Cindex is :" + str(average_Cindex/test_num))
