from datetime import datetime
from torch import nn


class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super(CrossAttentionFusion, self).__init__()
        self.cross_attn_gnn_to_gat = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.cross_attn_gat_to_gnn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, h_gnn, h_gat):
        # 变成 (batch, seq_len=1, embed_dim)
        h_gnn = h_gnn.unsqueeze(1)
        h_gat = h_gat.unsqueeze(1)

        # GNN 作为 Query，GAT 作为 Key 和 Value
        h_gnn_updated, _ = self.cross_attn_gnn_to_gat(h_gnn, h_gat, h_gat)

        # GAT 作为 Query，GNN 作为 Key 和 Value
        h_gat_updated, _ = self.cross_attn_gat_to_gnn(h_gat, h_gnn, h_gnn)

        # 融合（可以用均值、拼接或加权求和）
        h_fusion = (h_gnn_updated + h_gat_updated) / 2  # 平均
        # h_fusion = torch.cat([h_gnn_updated, h_gat_updated], dim=-1)  # 拼接
        # h_fusion = alpha * h_gnn_updated + (1 - alpha) * h_gat_updated  # 可学习加权求和

        return h_fusion.squeeze(1)  # 去掉 seq 维度


def log(*args):
    print(f'[{datetime.now()}]', *args)
    # 将信息保存到一个字符串变量中
    log_message = f'[{datetime.now()}] ' + ' '.join(map(str, args)) + '\n'

    # 将字符串写入到文件中
    with open('./Log/Log.txt', 'a') as f:
        f.write(log_message)
