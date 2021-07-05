import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from DeepBayesian.datasets import load_simulated_data
from DeepBayesian import CREN_create
from DeepBayesian import CREN
from DeepBayesian.datasets import survival_stats
from DeepBayesian.datasets import survival_df
from DeepBayesian.utils import concordance_index
from support import get_omic_data
from support import sort_data
from support import mkdir
from support import plot_curve
from support import multi_plot
from support import cal_pval
import torch
from math import log

#---------------------------------------------------------------------------------



##nn_config : is_random_bias True means using DBP methods False means using CoxNN



#---------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
nn_config = {
    "learning_rate": 0.0017,
    "learning_rate_decay": 0.999,
    "activation": 'selu',
    "epoch_num": 500,
    "skip_num": 5,
    "is_random_bias": True,
    "random_rate": 1.5,
    "censor_restore_rate": 0.2,
    "moving": 1.0,
    "L1_reg": 3.4e-5,
    "L2_reg": 1.2e-4,
    "optimizer": 'Adam',
    "dropout": 0.0,
    "dropout_keep_prob": 1.0,
    "hidden_layers": [18,8,1],
    "standardize": True,
    "batchnorm":False,
    "momentum": 0.8,
    "decay": 1.0,
    "alpha": 0,
    "seed": 1,
    "torch_seed": 1
}
input_nodes = 18
num_var = 3

average_Cindex = 0
test_num = 10
for i in range(test_num):
    train_data, train_risk = load_simulated_data(2000, N=800, method="gaussian", num_var=num_var, num_features=input_nodes, seed=i, censor_rate=0.5)
    test_data, test_risk= load_simulated_data(2000, N=2000, method="gaussian", num_var=num_var, num_features=input_nodes, seed=i+123, censor_rate=0.5)
    input_nodes = input_nodes
    surv_train = survival_df(train_data, t_col="t", e_col="e", label_col="Y")
    surv_test = survival_df(test_data, t_col="t", e_col="e", label_col="Y")
    Y_col = ["Y"]
    X_cols = [c for c in surv_train.columns if c not in Y_col]
    net = CREN_create(nn_config, input_nodes).to(device)
    model = CREN(nn_config, net)
    ori_train_X = surv_train[X_cols].values
    ori_train_Y = surv_train[Y_col].values
    ori_test_X = surv_test[X_cols].values
    ori_test_Y = surv_test[Y_col].values
    ori_idx, ori_train_X, ori_train_Y = sort_data(ori_train_X, ori_train_Y)
    model._train(device, ori_train_X, ori_train_Y)
    prediction = model._predict(device, ori_test_X)
    Cindex = concordance_index(ori_test_Y, -prediction)
    print("Cindex is : " + str(Cindex))
    average_Cindex += Cindex
print("Average Cindex is :" + str(average_Cindex / test_num))

