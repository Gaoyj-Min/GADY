import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms        

class Generator(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        # self.noise = torch.randn(self.node_emd_init.shape)
        self.node_emb_dim = 200
        self.time_emb_dim = 200
        self.dis = nn.Sequential(
            nn.Linear(2*self.node_emb_dim + self.time_emb_dim, 256),  # 输入特征数为784，输出为256
            nn.LeakyReLU(0.2),  # 进行非线性映射
            nn.Linear(256, 128),  # 进行一个线性映射
            nn.LeakyReLU(0.2),
            nn.Linear(128, 3),
            nn.Tanh()  # 也是一个激活函数，二分类问题中，
            # nn.Softmax(dim = 0)  # 也是一个激活函数，二分类问题中，
            # sigmoid可以班实数映射到【0,1】，作为概率值，
            # 多分类用softmax函数
        )

        
    def forward(self, batch):  # 前传函数
        self.noise = (torch.randn([batch.shape[0],2*self.node_emb_dim + self.time_emb_dim])-0.5)*45
        
        # self.noise = (torch.randn([batch.shape[0],2*self.node_emb_dim + self.time_emb_dim])-0.5)*
        x = self.dis(self.noise)
        x = (x+1)/2
        head = torch.floor(x[:,0]*(max(batch[:,0])-min(batch[:,0])) + min(batch[:,0]))
        tail = torch.floor(x[:,1]*(max(batch[:,1])-min(batch[:,1])) + min(batch[:,1]))
        time = torch.floor(x[:,2]*(max(batch[:,2])-min(batch[:,2])) + min(batch[:,2]))
        data = torch.hstack([head.reshape(-1,1),tail.reshape(-1,1),time.reshape(-1,1)])
        index = torch.sort(data[0:,2],descending=False)
        # data = data[np.lexsort([data.T[-1]])] 
        data = data[index.indices]
        data = data.type(torch.int64)
        return data  


