import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms


# class Discriminator(object):
#     def __init__(self, n_node, node_emd_init):
#         super(Discriminator, self).__init__()  # 继承初始化方法
        
#         self.img_size = img_size  # 图片尺寸，默认单通道灰度图
 
#         self.linear1 = nn.Linear(self.img_size[0] * self.img_size[1], 512)  # linear映射
#         self.linear2 = nn.Linear(512, 256)  # linear映射
#         self.linear3 = nn.Linear(256, 1)  # linear映射
#         self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)  # leakyrelu激活函数
#         self.sigmoid = nn.Sigmoid()  # sigmoid激活函数，将输出压缩至（0，1）       
        
#         self.n_node = n_node
#         self.node_emd_init = node_emd_init

#         self.embedding_matrix = torch.randn(self.node_emd_init.shape)
#         self.bias_vector = torch.zeros([self.n_node])

#         self.node_id = 0
#         self.node_neighbor_id = 0
#         self.reward = 0

#         self.node_embedding = torch.index_select(self.embedding_matrix, 0, self.node_id.long())
#         self.node_neighbor_embedding = torch.index_select(self.embedding_matrix, 0, self.node_neighbor_id.long())
#         self.bias = torch.index_select(self.bias_vector, 0, self.node_neighbor_id.long())
#         self.score = self.node_embedding*self.node_neighbor_embedding.sum(0) + self.bias
    
#     def forward(self, x):  # 前传函数
#         x = torch.flatten(x, 1)  # 输入图片从三维压缩至一维特征向量，(n,1,28,28)-->(n,784)
#         x = self.linear1(x)  # linear映射，(n,784)-->(n,512)
#         x = self.leakyrelu(x)  # leakyrelu激活函数
#         x = self.linear2(x)  # linear映射,(n,512)-->(n,256)
#         x = self.leakyrelu(x)  # leakyrelu激活函数
#         x = self.linear3(x)  # linear映射,(n,256)-->(n,1)
#         x = self.sigmoid(x)  # sigmoid激活函数
 
#         return x  # 返回图片真假的得分（置信度）


        

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
            nn.Linear(256, 256),  # 进行一个线性映射
            nn.LeakyReLU(0.2),
            nn.Linear(256, 3),
            nn.Tanh()  # 也是一个激活函数，二分类问题中，
            # nn.Softmax(dim = 0)  # 也是一个激活函数，二分类问题中，
            # sigmoid可以班实数映射到【0,1】，作为概率值，
            # 多分类用softmax函数
        )



        # self.n_node = n_node
        # self.node_emd_init = node_emd_init
        # self.embedding_matrix = torch.randn(self.node_emd_init.shape)
        # self.bias_vector = torch.zeros([self.n_node])

        # self.node_id = 0
        # self.node_neighbor_id = 0
        # self.reward = 0

        # self.all_score = torch.matmul(self.embedding_matrix,self.embedding_matrix.t()) + self.bias_vector
        # self.node_embedding = torch.index_select(self.embedding_matrix, 0, self.node_id)
        # self.node_neighbor_embedding = torch.index_select(self.embedding_matrix, 0, self.node_neighbor_id)

        # self.bias = torch.index_select(self.bias_vector, 0, self.node_neighbor_id)
        # self.score = self.node_embedding*self.node_neighbor_embedding.sum(0) + self.bias
        # self.prob = utils.clip_by_tensor(torch.sigmoid(self.score), 1e-5, 1)
        # nn.Linear(N_IDEAS,128),#全连接层
        # nn.ReLU(),
        # nn.Linear(128,ART_COMPONETS)#生成器把特征空间的128数据生成15维的
        
    def forward(self, batch):  # 前传函数
        self.noise = (torch.rand([batch.shape[0],2*self.node_emb_dim + self.time_emb_dim])-0.5)*30
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
        # sources_batch_neg, destinations_batch_neg, edge_time_neg = data[:,0], data[:,1], data[:,2]
        
        
        
        # x = torch.flatten(x, 1)  # 输入图片从三维压缩至一维特征向量，(n,1,28,28)-->(n,784)
        # x = self.linear1(x)  # linear映射，(n,784)-->(n,512)
        # x = self.leakyrelu(x)  # leakyrelu激活函数
        # x = self.linear2(x)  # linear映射,(n,512)-->(n,256)
        # x = self.leakyrelu(x)  # leakyrelu激活函数
        # x = self.linear3(x)  # linear映射,(n,256)-->(n,1)
        # x = self.sigmoid(x)  # sigmoid激活函数
 
        # return sources_batch_neg, destinations_batch_neg, edge_time_neg  # 返回图片真假的得分（置信度）
        return data  # 返回图片真假的得分（置信度）

        


# def artist_work():
#     a=np.random.uniform(1,2,size=BATCH_SIZE)[:,np.newaxis]
#     paints = a * np.power(PAINT_POINTS,2) + (a-1)#a*x^2+a-1 (a~(1,2))
#     paints = torch.from_numpy(paints).float()
#     return paints	#是一个64×15的张量，并且在两条线之间，


















