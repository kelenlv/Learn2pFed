import numpy as np
import torch.nn as nn

from os.path import join
import os
import torch
    
## Learn2pFed Network for synthetic dataset
class pFLNetLayer(nn.Module):
    def __init__(self, X, Y, n_clients, num, order=3, admm_iterations=10):
        super(pFLNetLayer, self).__init__()
        self.X = []
        self.Y = []
        self.n_clients = n_clients
        self.order = order
        self.fea = torch.tensor(self.order + 1).cuda() #Complement the 0th power term
        temp = 0
        for i in range(self.n_clients):
            self.X.append(self.features(X[temp:temp+num[i]]).cuda())
            self.Y.append(Y[temp:temp+num[i]].cuda())
            temp = temp+num[i]

        
        rho, p, tri_l, _ = self.init_learnable_param(self.Y,num)        
        self.rho = nn.Parameter(torch.Tensor(rho), requires_grad=True)
        self.eta = nn.Parameter(torch.Tensor(rho), requires_grad= True)
        self.gamma = nn.Parameter(torch.Tensor(rho), requires_grad=True)
        self.theta = nn.Parameter(torch.Tensor(rho), requires_grad= True)
        self.p = nn.Parameter(torch.Tensor(p), requires_grad = True)
        self.tri_l = nn.Parameter(torch.Tensor(tri_l).float(), requires_grad= True)

        
        self.var_init_layer = VariableInitLayer(self.fea, self.n_clients, self.tri_l, self.p)
        self.per_update_layer = PersonalizedUpdateLayer(self.rho,self.fea, self.n_clients, self.X, self.Y)
        self.aux_update_layer = AuxiliaryUpdateLayer(self.eta, self.tri_l, self.fea, self.n_clients)#
        self.glo_update_layer = GlobalUpdateLayer(self.p, self.gamma, self.n_clients)
        self.multiple_update_layer = MultipleUpdateLayer(self.theta, self.n_clients)
        layers = []
        layers.append(self.var_init_layer)
        for i in range(admm_iterations):
            layers.append(self.multiple_update_layer)
            layers.append(self.per_update_layer)
            layers.append(self.aux_update_layer)
            layers.append(self.glo_update_layer)
        self.pFL_net = nn.Sequential(*layers)

    def features(self, x): 
        return torch.cat([x ** i for i in range(0, self.order + 1)], 1)

    def init_learnable_param(self, Y,num):
        pt = 0
        p = []
        rho = []
        tri_l = []
        l_list = []
        l = 0.001*np.diag(np.random.rand(self.fea))
        for r  in range(self.n_clients):
            p.append(num[r]/np.sum(num)) 
            tri_l.append(l)#l
            rho.append(1e-2) 
        print(p)
        return rho, p, tri_l, l_list

    def forward(self, x):
        x = self.pFL_net(x)
        return x

## Learn2pFed Network for ELD dataset
class real_pFLNetLayer(nn.Module):
    def __init__(self, X, Y, n_clients,  admm_iterations=10):
        """
        Args:

        """
        super(real_pFLNetLayer, self).__init__()

        
        self.X = X
        self.Y = Y
        self.n_clients = n_clients
        self.fea = self.X[0].shape[1]         
        rho, p,  l = self.init_learnable_param(self.Y)        
        self.rho = nn.Parameter(torch.Tensor(rho), requires_grad=True)
        self.eta = nn.Parameter(torch.Tensor(rho), requires_grad= True)
        self.gamma = nn.Parameter(torch.Tensor(rho), requires_grad=True)
        self.theta = nn.Parameter(torch.Tensor(rho), requires_grad= True)
        self.p = nn.Parameter(torch.Tensor(p), requires_grad = True)
        self.l = nn.Parameter(torch.Tensor(l).type(torch.float64), requires_grad = True)

        
        self.var_init_layer = VariableInitLayer(self.fea, self.n_clients,  self.l, self.p)
        self.per_update_layer = PersonalizedUpdateLayer(self.rho,self.fea, self.n_clients, self.X, self.Y)
        self.aux_update_layer = AuxiliaryUpdateLayer(self.eta, self.l, self.fea, self.n_clients)#
        self.glo_update_layer = GlobalUpdateLayer(self.p, self.gamma, self.n_clients)
        self.multiple_update_layer = MultipleUpdateLayer(self.theta, self.n_clients)

        layers = []
        layers.append(self.var_init_layer)
        for i in range(admm_iterations):
            layers.append(self.multiple_update_layer)
            layers.append(self.per_update_layer)
            layers.append(self.aux_update_layer)
            layers.append(self.glo_update_layer)
        self.pFL_net = nn.Sequential(*layers)

    def init_learnable_param(self, Y):
        p = []
        rho = []
        l_list = []
        np.random.seed(12)
        l = 1*np.random.rand(self.fea) 
        pt = 0
        for i in range(self.n_clients):
            pt = pt + np.size(Y[i],0)
        for r  in range(self.n_clients):
            p.append(np.size(Y[r],0)/pt)            
            l_list.append(l)
            rho.append(1e-0) #0.01
        return rho, p,  l_list
    def forward(self, x):
        x = self.pFL_net(x)
        return x

## Initialization layer
class VariableInitLayer(nn.Module):
    def __init__(self, fea, n_clients, tri_l, p):
        super(VariableInitLayer,self).__init__()
        self.fea = fea
        self.n_clients = n_clients
        self.tri_l = tri_l
        self.p = p

    def forward(self, x):
        ##initialization
        v = []
        w = torch.randn(self.fea,1).cuda()
        z = []
        alpha = []
       
        for r  in range(self.n_clients):
            v.append(torch.randn(self.fea,1).cuda())
            z.append((v[r] - w).cuda())
            alpha.append(torch.zeros(self.fea, 1).cuda()) 
        
        
        # define data dict
        pFL_data = dict()
        pFL_data['input'] = x
        pFL_data['variable_v'] = v
        pFL_data['variable_z'] = z
        pFL_data['variable_alpha'] = alpha
        pFL_data['variable_w'] = w
        pFL_data['param_tri_l'] = self.tri_l
        pFL_data['param_p'] = self.p
        return pFL_data

## Personalized variable update layer
class PersonalizedUpdateLayer(nn.Module):
    def __init__(self, rho, fea, n_clients, X, Y):
        super(PersonalizedUpdateLayer,self).__init__()
        self.rho = rho
        self.fea = fea
        self.n_clients = n_clients
        self.X = X
        self.Y = Y

    def forward(self, x):
        w = x['variable_w']
        v = x['variable_v']
        z = x['variable_z']
        alpha = x['variable_alpha']
        input = x['input']

        
        temp = []
        for r in range(self.n_clients):
            tt = torch.transpose(self.X[r].squeeze(),0,1)@self.X[r].squeeze()+(self.rho[r]*torch.eye(self.fea).cuda()).float()
            temp.append(self.rho[r]*(w+z[r]+alpha[r]) + torch.transpose(self.X[r].squeeze(),0,1)@self.Y[r])
            v[r] = torch.linalg.pinv(tt)@temp[r]
        x['variable_v'] = v
        return x


## Auxiliary variable update layer
class AuxiliaryUpdateLayer(nn.Module):
    def __init__(self, eta, tri_l, fea, n_clients):
        super(AuxiliaryUpdateLayer,self).__init__()
        self.eta = eta
        self.tri_l = tri_l
        self.fea = fea
        self.n_clients = n_clients

        

    def forward(self, x):
        w = x['variable_w']
        v = x['variable_v']
        z = x['variable_z']
        alpha = x['variable_alpha']
        
        
        for r in range(self.n_clients):
            z[r] = (self.eta[r]*(torch.linalg.pinv((self.tri_l[r])+self.eta[r]*torch.eye(self.fea).cuda())).float()@(v[r]-w-alpha[r])).float()
        

        x['variable_z'] = z
        return x

## Global variable update layer
class GlobalUpdateLayer(nn.Module):
    def __init__(self, p, gamma, n_clients):
        super(GlobalUpdateLayer,self).__init__()
        self.p = p
        self.gamma = gamma
        self.n_clients = n_clients

    def forward(self, x):
        v = x['variable_v']
        z = x['variable_z']
        alpha = x['variable_alpha']
     
        c = []
        vec = []
        t1 = 0
        t2 = 0
        
        for r in range(self.n_clients):
            c.append(self.p[r]*self.gamma[r])
            vec.append(v[r]-z[r]-alpha[r])
            t1 = t1 + c[r]*vec[r]
            t2 = t2 + c[r]

        t1.clone().detach().requires_grad_(True).float()
        t2.clone().detach().requires_grad_(True).float()
        x['variable_w'] = t1/t2
        return x

## Multiplier variable update layer
class MultipleUpdateLayer(nn.Module):
    def __init__(self, theta, n_clients):
        super(MultipleUpdateLayer,self).__init__()
        self.theta = theta
        self.n_clients = n_clients

    def forward(self, x):
        v = x['variable_v']
        z = x['variable_z']
        w = x['variable_w']
        alpha = x['variable_alpha']

        
        for r in range(self.n_clients):
            alpha[r] = alpha[r] + self.theta[r]*(z[r]-v[r] + w)
        
        x['variable_alpha'] = alpha 
        return x
        
