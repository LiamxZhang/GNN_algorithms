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
        num_neighbors = adj_matrix.sum(dim=-1,keepdims=True) #各节点的邻居数
        node_feats = self.projection(node_feats)# 将特征转化为消息
        #各邻居节点消息求和并求平均
        node_feats = torch.bmm(adj_matrix,node_feats)
        node_feats = node_feats / num_neighbors
        
        return node_feats
    
if __name__ == "__main__":
    node_feats = torch.arange(8,
    dtype=torch.float32).view(1,4,2)
    adj_matrix = torch.Tensor([[[1,1,0,0],
                [1,1,1,1],
                [0,1,1,1],
                [0,1,1,1]]])
    print("节点特征：\n",node_feats)
    print("添加自链接的邻接矩阵：\n",adj_matrix)

    layer = GCNLayer(c_in=2, c_out=2)
    # 初始化权重矩阵
    layer.projection.weight.data = torch.Tensor([[1., 0.], [0., 1.]])
    layer.projection.bias.data = torch.Tensor([0., 0.])
    
    # 将节点特征和添加自连接的邻接矩阵输入 GCN 层
    with torch.no_grad():
        out_feats = layer(node_feats, adj_matrix)
    
    print("节点特征:\n", node_feats)
    print("添加自连接的邻接矩阵:\n", adj_matrix)
    print("节点输出特征:\n", out_feats)

