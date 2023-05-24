import os
import torch
import random
import argparse
import numpy as np
from typing import Iterator, Tuple, Union
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import torch.utils.data as Data
from torch.optim.optimizer import required
import copy
from collections import OrderedDict, defaultdict
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, r2_score, explained_variance_score
from load_ELD import load_ELD
from Layers import pFLNetLayer, real_pFLNetLayer
import setproctitle

def seed(seed_value = 42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

## generating synthetic dataset
def gen_synthetic(n_clients,a,n_coefficients, is_reverse):
    seed(42)
    sizee = 1000
    x = np.random.uniform(0,1,size=sizee)#1,1000
    x = np.sort(x)
    y = a[0] + a[1]*x + a[2]*x**2 + a[3]*x**3
    dd = int(len(x)/n_clients)
    aa = np.random.rand(n_clients)
    
    num = []
    for i in range(n_clients):
        num.append(int(aa[i]/aa.sum()*sizee))

    
    X_gt = []
    Y_gt = []
    X = []
    Y = []
    temp = 0
    for i in range(n_clients):
        X_gt.append([])
        Y_gt.append([])
        dd = num[i]  
        X.append(x[temp:temp+dd])
        Y.append(y[temp:temp+dd])
        temp = temp+dd
       
    ## local gt
    plt.figure()
    b_list = 100*np.random.uniform(0-0.1,0+0.1,n_clients)
    for i in range(n_clients):
        rot(X[i], Y[i], b_list[i],a,n_coefficients, is_reverse)
    for i in range(n_clients):
        plt.scatter(X[i], Y[i])
    
    plt.title('local gt')
    plt.savefig('lgt.png', bbox_inches = 'tight')

    ## local gt backup
    plt.figure()     
    for i in range(n_clients):
        for j in range(len(X[i])):
            X_gt[i].append(X[i][j])
            Y_gt[i].append(Y[i][j])
    for i in range(n_clients):
        plt.scatter(X_gt[i], Y_gt[i])


    ## local nosiy data
    plt.figure()
    for i in range(n_clients):
        gauss_noisy(X[i], Y[i], 0.1 , 0)#10+10*i
    for i in range(n_clients):
        plt.scatter(X[i], Y[i])
    plt.title('local data')
    plt.savefig('ld.png', bbox_inches = 'tight')
    return X, Y, X_gt, Y_gt, b_list, num
## 1. rotation by a 
## 2. translation by b
def rot(x, y, b,a,n_coefficients, is_reverse):
    print('is_reverse:', is_reverse) #False
    if is_reverse:
        for i in range(len(x)):
            if n_coefficients == 1:
                y[i] = a[0]+b+ (a[1])*x[i] + (a[2])*x[i]**2 + (a[3])*np.power(x[i], 3)
            elif n_coefficients == 2:
                y[i] = a[0]+b+ (a[1]+b)*x[i] + (a[2])*x[i]**2 + (a[3])*np.power(x[i], 3)
            elif n_coefficients == 3:
                y[i] = a[0]+b + (a[1]+b)*x[i] + (a[2]+b)*x[i]**2 + (a[3])*np.power(x[i], 3)
            else:
                print('orders failure!')
    else: #default
        print('in defualt')
        for i in range(len(x)):
            if n_coefficients == 1: ## s1 in paper : three are same
                y[i] = a[0]+ (a[1])*x[i] + (a[2])*x[i]**2 + (a[3]+b)*np.power(x[i], 3)
            elif n_coefficients == 2: ## s2 in paper: two are same
                y[i] = a[0]+ (a[1])*x[i] + (a[2]+b)*x[i]**2 + (a[3]+b)*np.power(x[i], 3)
            elif n_coefficients == 3: ## s3 in paper: one is same  # harder
                y[i] = a[0]+ (a[1]+b)*x[i] + (a[2]+b)*x[i]**2 + (a[3]+b)*np.power(x[i], 3)
            else:
                print('orders failure!')

## 3. add noise
def gauss_noisy(x, y, sigma, mu=0):
    for i in range(len(x)):
        y[i] += random.gauss(mu, sigma)
## classical federated aggregation
def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg
## complement the 0th power term
def features(x, order=3): 
    return torch.cat([x ** i for i in range(0,order+1)], 1)

## splitting ELD dataset
def split_data(x,y,split_ratio=0.9):
    train_size=int((y.shape[0])*split_ratio)
    test_size=(y.shape[0])-train_size

    x_data=Variable(torch.Tensor(np.array(x)))
    y_data=Variable(torch.Tensor(np.array(y)))

    x_train=Variable(torch.Tensor(np.array(x[0:train_size])))
    y_train=Variable(torch.Tensor(np.array(y[0:train_size])))
    y_test=Variable(torch.Tensor(np.array(y[train_size:len(y)])))
    x_test=Variable(torch.Tensor(np.array(x[train_size:len(x)])))

    print('x_data.shape,y_data.shape,x_train.shape,y_train.shape,x_test.shape,y_test.shape:\n{}{}{}{}{}{}'
    .format(x_data.shape,y_data.shape,x_train.shape,y_train.shape,x_test.shape,y_test.shape))

    return x_data,y_data,x_train,y_train,x_test,y_test

seed()
parser = argparse.ArgumentParser(description=' main ')
parser.add_argument('--batch_size', default = 1, type=int, help='batch size')
parser.add_argument('--n_clients', default=5, type=int, help='number of clients')
parser.add_argument('--fed_epoch', default=500, type=int, help='number of federation epochs')
parser.add_argument('--outer_epoch', default=1, type=int, help='repeated exps')
parser.add_argument('--local_epoch', default=10, type=int, help='local epochs')#50
parser.add_argument('--alg', default= 'Learn2pFed', type=str, help='alg_list = ["Learn2pFed"]')
parser.add_argument('--n_coefficients', default = 1, type=int,  help='different number of coefficients')
parser.add_argument("--is_finetune", action="store_true",help="is finetuning state or not")# default False
parser.add_argument("--is_reverse", action="store_true",help="is finetuning state or not")# default False
parser.add_argument("--tune_epoch", default = 500, type=int)
parser.add_argument("--dataset", default = 'synthetic', type=str, help='synthetic / ELD / ')
parser.add_argument("--admm_iter", default = '10', type=int, help='admm_iterations')

args = parser.parse_args()
## data prepare
X_train = []
Y_train = []
X_test = []
Y_test = []
train_data = []
train_loader = []
test_data = []
test_loader = []
train_loss_all = []
if args.dataset == 'synthetic':
    a = [0, -6, 18, -12]#,1        
    X, Y,  X_gt, Y_gt, b_list, num = gen_synthetic(args.n_clients,a, args.n_coefficients, args.is_reverse)
    for i in range(args.n_clients):
        X_train.append(torch.from_numpy(X[i]).reshape(-1,1).float())
        save_X.append(torch.from_numpy(X[i]).reshape(-1,1).float())
        X_train[i] = features(X_train[i], order=1).cuda()
        Y_train.append(torch.from_numpy(Y[i]).reshape(-1,1).float())
        train_data.append(Data.TensorDataset(X_train[i], Y_train[i]))
        train_loader.append(Data.DataLoader(dataset = train_data[i], batch_size = 1, 
                                shuffle = True, num_workers = 1))
        train_loss_all.append([]) 
    Ytr_gt = Y_train
elif args.dataset == 'ELD':
    num = []
    n_iters = 5000
    batch_size = 64
    
    if args.n_clients == 5:
        x, y, ex_x, ex_y = load_ELD(args.n_clients)
        _,_,_,_,ex_x_test,ex_y_test = split_data(ex_x, ex_y) #non-participating clients
    else:
        x, y = load_ELD(args.n_clients)

    for i in range(args.n_clients):
        x_data,y_data,x_train,y_train,x_test,y_test = split_data(x[i],y[i])
        X_train.append(x_train.cuda())
        Y_train.append(y_train.cuda())
        X_test.append(x_test.cuda())
        Y_test.append(y_test)
        num.append(x_train.shape[0])
        train_data.append(Data.TensorDataset(x_train,y_train))
        test_data.append(Data.TensorDataset(x_test,y_test))
        train_loader.append(Data.DataLoader(dataset = train_data[i], batch_size = 64, 
                                shuffle=False,drop_last=True))
        test_loader.append(Data.DataLoader(dataset = test_data[i], batch_size = 64, 
                                shuffle=False,drop_last=True))
    Ytr_gt = Y_train
    Yts_gt = Y_test
else:
    print('Data Not Prepared!')

    
if torch.cuda.is_available():
    if args.dataset == 'ELD':
        seq_length=3 
        input_size= 1
        num_layers=2
        hidden_size=12
        batch_size= 1 
        output_size=1
        if args.alg == 'Learn2pFed':
            net = real_pFLNetLayer(X_train, Y_train, args.n_clients,   admm_iterations=args.admm_iter).cuda()
    else:
        if args.alg == 'Learn2pFed':
            net = pFLNetLayer(X_train, Y_train, args.n_clients, num, order=1, admm_iterations=args.admm_iter).cuda()


record_List = []
tr_record_List = []
ts_record_List = []

w_glob = net.state_dict()

loss_func = nn.MSELoss()
net_local = []

print('in %s training ~~' % args.alg)



if args.alg == 'Learn2pFed':
    param_list = dict( admm_iterations = args.admm_iter)

## concanating target Y
if args.alg == 'Learn2pFed':
    target = Y_train[0].reshape(-1,1)
    for i in range(args.n_clients):
        if i>0:
            target = torch.cat((target,Y_train[i].reshape(-1,1)), dim=0)

for outer in range(args.outer_epoch):
    total_loss_org_List = []
    for epoch in range(args.fed_epoch):
        if args.alg == 'Learn2pFed':
            net.train()
            optimizer = torch.optim.SGD(net.parameters(), lr=1e-2)
            if epoch > 60:
                optimizer = torch.optim.SGD(net.parameters(), lr = 0.001)
            total_loss_org = 0
            ## train
            output = net(X_train)
            optimizer.zero_grad()
            y_pred = X_train[0].squeeze()@output['variable_v'][0]
            for i in range(args.n_clients):
                if i > 0:
                    y_pred = torch.cat((y_pred, X_train[i].squeeze()@output['variable_v'][i]), dim = 0)  

            loss = loss_func(y_pred.cuda(), target.cuda())
            
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss_org += loss.data.item()
            
            total_loss_org_List.append( total_loss_org)
            if epoch > 0 :
                if np.linalg.norm(total_loss_org_old-total_loss_org)< 1e-8:
                    break
            total_loss_org_old = total_loss_org

    ## test
    client = []
    if args.dataset == 'ELD':
        tr_predict_list = []
        ts_predict_list = []
        tr_record = []
        ts_record = []
        tr_predict=[]
        ts_predict = []
        for i in range(args.n_clients):
            tr_predict.append([])
            ts_predict.append([])
            plt.figure()
            if torch.cuda.is_available():
                with torch.no_grad():
                    if args.alg == 'Learn2pFed':
                        tr_predict[i] = X_train[i].squeeze()@output['variable_v'][i]
                        tr_predict[i] = tr_predict[i].data.cpu().numpy()   
                        tr_predict_list.append(tr_predict[i])
                    
                        ts_predict[i] = X_test[i].squeeze()@output['variable_v'][i]
                        ts_predict[i] = ts_predict[i].data.cpu().numpy()
                        ts_predict_list.append(ts_predict[i])
            
            plt.plot(ts_predict_list[i][0:2000], label = 'prediction', linewidth=2)
            plt.plot(Y_test[i][0:2000],'k-', label = 'gt', linewidth=2)
            plt.legend()
            plt.xlabel('time')
            plt.ylabel('power consumption')
            plt.savefig('Learn2pFed_ELD_%s_%d.png'%(args.alg, i))
        if args.n_clients == 5:
            for i in range(args.n_clients):
                plt.figure()
                if torch.cuda.is_available():
                    with torch.no_grad():
                        if args.alg == 'Learn2pFed':
                            ex_ts_predict = ex_x_test.squeeze().cuda()@output['variable_v'][i]
                            ex_ts_predict = ex_ts_predict.data.cpu().numpy()
                    
                plt.plot(ex_ts_predict[0:2000], label = 'prediction')
                plt.plot(ex_y_test[0:2000],'k-', label = 'gt')
                plt.legend()
                plt.xlabel('time')
                plt.ylabel('power consumption')
                plt.savefig('Learn2pFed_ELD_ex_%s_%d.png'%(args.alg, i))
    elif args.dataset == 'synthetic':
        if args.is_finetune:
            print('debug: in finetuning')
            record = []
            tr_record = []
            plt.figure()
            for i in range(args.n_clients):
                client.append(copy.deepcopy(net))
                optimizer = torch.optim.SGD(client[i].parameters(), lr = 1e-2)
                if torch.cuda.is_available():
                        inputs = Variable(X_train[i]).cuda()
                        target = Variable(Y_train[i]).cuda()
                else:
                    inputs = Variable(X_train[i])
                    target = Variable(Y_train[i])
                for epoch in range(args.tune_epoch):
                    out = client[i](inputs)
                    loss = loss_func(out, target)

                    optimizer.zero_grad()  
                    loss.backward()
                    optimizer.step()

                client[i].eval()
                if torch.cuda.is_available():
                    predict = client[i](Variable(X_train[i]).cuda())
                    predict = predict.data.cpu().numpy()
                else:
                    predict = client[i](Variable(X_train[i]))
                    predict = predict.data.numpy()
                

                tr_record.append(F.mse_loss(torch.tensor(predict), torch.tensor(Y_gt[i]).reshape(-1,1).float()))
                print('mse:', i, F.mse_loss(torch.tensor(predict), torch.tensor(Y_gt[i]).reshape(-1,1).float()))
                plt.plot(X[i], predict,  linewidth=2)
                plt.plot(X_gt[i], Y_gt[i],'k--', linewidth=2)
            tr_record_List.append(np.mean(np.array(tr_record)))
        else:        
            print('test: NOT in finetuning')
            plt.figure()
            record = []
            tr_record = []
            ts_record = []
            tr_predict=[]
            
            for i in range(args.n_clients):
                tr_predict.append([])
                if torch.cuda.is_available():
                    with torch.no_grad():
                        if args.alg == 'Learn2pFed':
                            tr_predict[i] = X_train[i]@output['variable_v'][i]
                            tr_predict[i] = tr_predict[i].data.cpu().numpy()                           
                if args.alg == 'Learn2pFed':
                    pass
                    tr_record.append(F.mse_loss(torch.tensor(tr_predict), torch.tensor(Ytr_gt[i]).reshape(-1,1).float()))
                    print('tr_mse:', i, F.mse_loss(torch.tensor(tr_predict), torch.tensor(Ytr_gt[i]).reshape(-1,1).float()))
                    plt.plot(X[i], tr_predict,  linewidth=2)
                    plt.plot(X_gt[i], Y_gt[i],'k--', linewidth=2)
                    plt.plot(total_loss_org_List)
                    plt.savefig('Learn2pFed_%s_s%d_admm%d.png'%(args.dataset , args.n_coefficients, args.admm_iter), bbox_inches = 'tight')
                    np.save('loss_list_%s_s%d_admm%d.npy'%(args.dataset, args.n_coefficients, args.admm_iter), total_loss_org_List)
                     
            tr_record_List.append(np.mean(np.array(tr_record)))
            ts_record_List.append(np.mean(np.array(ts_record)))
            
    

print(tr_record_List)
print('tr_mean:', format(np.mean(np.array(tr_record_List)), '.4f'), 'tr_std:',format(np.std(np.array(tr_record_List),ddof=1), '.4f'))

print(ts_record_List)
print('ts_mean:', format(np.mean(np.array(ts_record_List)), '.4f'), 'ts_std:',format(np.std(np.array(ts_record_List),ddof=1), '.4f'))
if not args.is_finetune:
    print('setting:', param_list, args.tune_epoch)
else:
    print('setting:', param_list)