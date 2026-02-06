import torch
import torch.nn as nn
 
class GCNLayer(nn.Module):
 
    def __init__(self,c_in,c_out):
        """
        Inputs:
        :param c_in: 输入特征
        :param c_out: 输出特征
        """
        super().__init__()
        self.projection = nn.Linear(c_in,c_out); #线性层
        
    def forward(self,node_feats,adj_matrix):
        """
        输入
        :param node_feats: 节点特征表示，大小为[batch_size,num_nodes,c_in]
        :param adj_matrix: 邻接矩阵：[batch_size,num_nodes,num_nodes]
        :return:
        """
        num_neighbors = adj_matrix.sum(dim=-1,keepdims=True)#各节点的邻居数
        node_feats = self.projection(node_feats)#将特征转化为消息
        #各邻居节点消息求和并求平均
        node_feats = torch.bmm(adj_matrix,node_feats)
        node_feats = node_feats / num_neighbors
        return node_feats