import torch
import torch.nn as nn
import torch.nn.functional as F
 
class GATLayer(nn.Module):
 
    def __init__(self,c_in,c_out,
                num_heads=1, concat_heads=True, alpha=0.2):
        """
        :param c_in: 输入特征维度
        :param c_out: 输出特征维度
        :param num_heads: 多头的数量
        :param concat_heads: 是否拼接多头计算的结果
        :param alpha: LeakyReLU的参数
        :return:
        """
        super().__init__()
        self.num_heads = num_heads
        self.concat_heads = num_heads
        if self.concat_heads:
            assert c_out % num_heads ==0,"输出特征数必须是头数的倍数！"
            c_out = c_out // num_heads
 
        #参数
        self.projection = nn.Linear(c_in,c_out*num_heads) #有几个头，就需要将c_out扩充几倍
        self.a = nn.Parameter(torch.Tensor(num_heads,2*c_out)) #用于计算注意力的参数，由于对两节点拼接后的向量进行操作，所以2*c_out
        self.leakrelu = nn.LeakyReLU(alpha) #激活层
 
        #参数初始化
        nn.init.xavier_uniform_(self.projection.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
 
    def forward(self,node_feats,adj_matrix,print_attn_probs=False):
        """
        输入：
        :param self:
        :param node_feats: 节点的特征表示
        :param adj_matrix: 邻接矩阵
        :param print_attn_probs: 是否打印注意力
        :return:
        """
        batch_size,num_nodes = node_feats.size(0),node_feats.size(1)

        #将节点初始输入进行权重运算
        node_feats = self.projection(node_feats)
        #扩展出多头数量的维度
        node_feats = node_feats.view(batch_size,num_nodes,self.num_heads,-1)

        # 获取所有顶点对拼接而成的特征向量 a_input
        edges = adj_matrix.nonzero(as_tuple=False)  # 返回所有邻接矩阵中值不为 0 的 index，即所有连接的边对应的两个顶点
        node_feats_flat = node_feats.view(batch_size * num_nodes, self.num_heads, -1)  # 将所有 batch_size 的节点拼接

        edge_indices_row = edges[:, 0] * batch_size + edges[:, 1]  # 获取边对应的第一个顶点 index
        edge_indices_col = edges[:, 0] * batch_size + edges[:, 2]  # 获取边对应的第二个顶点 index

        a_input = torch.cat([
        torch.index_select(input=node_feats_flat, index=edge_indices_row, dim=0), # 基于边对应的第一个顶点的 index 获取其特征值
            torch.index_select(input=node_feats_flat, index=edge_indices_col, dim=0)  # 基于边对应的第二个顶点的 index 获取其特征值
        ], dim=-1)  # 两者拼接

        # 基于权重 a 进行注意力计算
        attn_logits = torch.einsum('bhc,hc->bh', a_input, self.a)
        # LeakyReLU 计算
        attn_logits = self.leakrelu(attn_logits)

        # 将注意力权转换为矩阵的形式
        attn_matrix = attn_logits.new_zeros(adj_matrix.shape + (self.num_heads,)).fill_(-9e15)
        attn_matrix[adj_matrix[..., None].repeat(1, 1, 1, self.num_heads) == 1] = attn_logits.reshape(-1)

        # Softmax 计算转换为概率
        attn_probs = F.softmax(attn_matrix, dim=2)
        if print_attn_probs:
            print("注意力权重:\n", attn_probs.permute(0, 3, 1, 2))
        # 对每个节点进行注意力加权相加的计算
        node_feats = torch.einsum('bijh,bjhc->bihc', attn_probs, node_feats)

        # 根据是否将多头的计算结果拼接与否进行不同操作
        if self.concat_heads:  # 拼接
            node_feats = node_feats.reshape(batch_size, num_nodes, -1)
        else:  # 平均
            node_feats = node_feats.mean(dim=2)

        return node_feats 

if __name__ == "__main__":
    layer = GATLayer(2, 2, num_heads=2)
    layer.projection.weight.data = torch.Tensor([[1., 0.], [0., 1.]])
    layer.projection.bias.data = torch.Tensor([0., 0.])
    layer.a.data = torch.Tensor([[-0.2, 0.3], [0.1, -0.1]])
    node_feats = torch.arange(8, dtype=torch.float32).view(1, 4, 2)
    adj_matrix = torch.Tensor([[[1, 1, 0, 0],
                                        [1, 1, 1, 1],
                                        [0, 1, 1, 1],
                                        [0, 1, 1, 1]]])
    with torch.no_grad():
        out_feats = layer(node_feats, adj_matrix, print_attn_probs=True)
    
    
    print("节点特征:\n", node_feats)
    print("添加自连接的邻接矩阵:\n", adj_matrix)
    print("节点输出特征:\n", out_feats)