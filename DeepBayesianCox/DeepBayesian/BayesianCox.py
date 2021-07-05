import pandas as pd
import numpy as np
import torch
from scipy.stats import norm
from .utils import _prepare_surv_data
from .utils import concordance_index
import numpy as np

def _sort(X, Y):
    T = -np.abs(np.squeeze((Y)))
    sorted_idx = np.argsort(T)
    return X[sorted_idx], Y[sorted_idx]

def random_bias(train_X, train_t, random_rate, censor_restore_rate):
    ## train_t has been sorted
    train_X = train_X.numpy()
    train_t = train_t.numpy().squeeze()

    t_E = train_t.copy()
    t_E[train_t > 0] = 1
    t_E[train_t <= 0] = 0
    ## rank
    rank_t = np.arange(train_t.size, 0, -1)
    random_num = np.random.normal(0, random_rate, size=train_t.size)
    rank = rank_t + random_num
    t_E[random_num>=(norm.ppf(1 - censor_restore_rate)*random_rate)] = 1
    ##resort
    t_value = rank.copy()
    t_value[t_E == 0] *= -1
    t_value = t_value[:, None]
    train_X,t_value = _sort(train_X, t_value)
    return torch.from_numpy(train_X), torch.from_numpy(t_value)


class CREN_create(torch.nn.Module):
    def __init__(self, nn_config, input_nodes):
        super(CREN_create, self).__init__()
        torch.manual_seed(nn_config["torch_seed"])
        torch.set_printoptions(profile="full")
        self.hidden_layers = nn_config["hidden_layers"]
        self.nn_config = nn_config
        self.input_nodes = input_nodes
        self.layer_set = torch.nn.ModuleList()
        self.batch_set = torch.nn.ModuleList()
        input_dim = input_nodes
        if self.nn_config["batchnorm"]:
            self.batch_set.append(
                torch.nn.BatchNorm1d(input_dim, momentum=self.nn_config["momentum"]))
        for i, output_dim in enumerate(self.hidden_layers):
            self.layer_set.append(torch.nn.Linear(input_dim, output_dim))
            if self.nn_config["batchnorm"]:
                self.batch_set.append(
                    torch.nn.BatchNorm1d(output_dim, momentum=self.nn_config["momentum"]))
            input_dim = output_dim
    def activation_func(self, x):
        if self.nn_config["activation"] == "relu":
            return torch.nn.functional.relu(x)
        elif self.nn_config["activation"] == "selu":
            return torch.nn.functional.selu(x)
        elif self.nn_config["activation"] == "tanh":
            return torch.nn.functional.tanh(x)
        elif self.nn_config["activation"] == "sigmoid":
            return torch.nn.functional.sigmoid(x)
    def forward(self, x):
        if self.nn_config["batchnorm"]:
            x = self.batch_set[0](x)
        for i in range(len(self.hidden_layers)):
            if i == 2: decode_pred = x
            x = self.layer_set[i](x)
            if self.nn_config["batchnorm"]:
                x = self.batch_set[i + 1](x)
            if i != len(self.hidden_layers)-1:
                x = self.activation_func(x)  # relu / selu
                x = torch.nn.Dropout(self.nn_config["dropout"])(x)
        y_predict = x
        return y_predict

class DeepCox_LossFunc(torch.nn.Module):
    def __init__(self, nn_config):
        super(DeepCox_LossFunc, self).__init__()
        self.nn_config = nn_config
        return
    def forward(self, y_predict, t):
        y_pred_list = y_predict.view(-1)
        y_pred_exp = torch.exp(y_pred_list)
        t_list = t.view(-1)
        t_E = torch.gt(t_list,0)
        y_pred_cumsum = torch.cumsum(y_pred_exp, dim=0)
        y_pred_cumsum_log = torch.log(y_pred_cumsum)
        # set the same survival time patients the same denominator
        # length = len(t_T)
        # for i in range(1,length):
        #     if torch.equal(t_T[length-i-1],t_T[length-i]):
        #         y_pred_cumsum[length-i-1] = y_pred_cumsum[length-i]
        ### loss combine
        loss1 = -torch.sum(y_pred_list.mul(t_E))
        loss2 = torch.sum(y_pred_cumsum_log.mul(t_E))
        loss = (loss1 + loss2)/torch.sum(t_E)
        return loss



class CREN():
    def __init__(self,nn_config, net):
        self.nn_config = nn_config
        self.net = net
        self._optimize()
    def save_fea(self, x):
        mean_set = []
        std_set = []
        #print(len(x[0]))
        for j in range(len(x[0])):
            mean = np.mean(x[:, j])
            std = np.std(x[:, j])
            mean_set.append(mean)
            if std == 0:
                std = 1
            std_set.append(std)
        self.mean_set = mean_set
        self.std_set = std_set
    def standardize(self, x):
        for j in range(len(x[0])):
            x[:, j] = (x[:, j] - self.mean_set[j]) / self.std_set[j]
        return x
    def _optimize(self):
        if self.nn_config["optimizer"] == "Adam":
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.nn_config["learning_rate"], weight_decay = self.nn_config['L2_reg'])
        torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[0, 1000], gamma=self.nn_config["learning_rate_decay"])
    def _loss(self):
        return DeepCox_LossFunc(self.nn_config)

    def _train(self, device, X, Y):
        if self.nn_config["standardize"]:
            self.save_fea(X)
            X = self.standardize(X)
        self.X = X
        self.Y = Y
        X = X.astype(np.float32)
        Y = Y.astype(np.float32)


        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)
        cal_loss = self._loss()
        self.net.train()
        for i in range(self.nn_config["epoch_num"]):
            np.random.seed(i)
            if self.nn_config["is_random_bias"]:
                pred = self.net(X.to(device)).detach().cpu().numpy()
                X_train, Y_train = random_bias(X, Y, self.nn_config["random_rate"], self.nn_config["censor_restore_rate"])
            else:
                X_train = X
                Y_train = Y
            X_train = X_train.to(device)
            Y_train = Y_train.to(device)
            y_predict = self.net(X_train)
            loss = cal_loss(y_predict, Y_train)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if i % self.nn_config["skip_num"] == 0:
                #print(loss.item())
                self.net.eval()
                t_y_predict = self.net(X.to(device))
                y_predict_save = t_y_predict.detach()
                self.net.train()
        return

    def _predict(self, device, input_x):
        if self.nn_config["standardize"]:
            input_x = self.standardize(input_x)
        self.net.eval()
        input_x = input_x.astype(np.float32)
        input_x = torch.from_numpy(input_x)
        pred = self.net(input_x.to(device))
        pred = pred.view(-1).detach()
        self.net.train()
        return np.exp(pred.cpu().numpy())



