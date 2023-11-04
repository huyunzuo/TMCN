import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from model.models import FewShotModel

# Note: As in Protonet, we use Euclidean Distances here, you can change to the Cosine Similarity by replace
#       TRUE in line 30 as self.args.use_euclidean

import torch
from einops import rearrange, repeat
from torch import nn, einsum
from torch.nn.init import trunc_normal_


class LayerNorm(nn.Module):  # layernorm, but done in the channel dimension #1
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, bias=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size, padding=padding, groups=dim_in, stride=stride, bias=bias),
            nn.BatchNorm2d(dim_in),
            nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=bias),
        )

    def forward(self, x):
        return self.net(x)


class AttentionCNN(nn.Module):
    def __init__(self, in_dim, dim_head=64, heads=8, residual_mode=True):
        super().__init__()
        proj_kernel = 3
        kv_proj_stride = 2
        inner_dim = dim_head * heads
        padding = proj_kernel // 2
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_q = DepthWiseConv2d(in_dim, inner_dim, proj_kernel, padding, stride=1)
        self.to_kv = DepthWiseConv2d(in_dim, inner_dim * 2, proj_kernel, padding, stride=kv_proj_stride)
        self.residual_mode = residual_mode
        # print('self.residual_mode: ', residual_mode)
        self.norm = []
        for _ in range(heads):
            self.norm.append(nn.Sequential(
                LayerNorm(dim_head)))
        self.norm = nn.ModuleList(self.norm)
        if self.residual_mode:
            if in_dim != dim_head:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_dim, dim_head, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(dim_head),
                )
            else:
                self.downsample = nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        if self.residual_mode:
            residual = self.downsample(x)
        shape = x.shape
        b, n, _, y, h = *shape, self.heads
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=1))
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> (b h) (x y) d', h=h), (q, k, v))
        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = self.attend(dots)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) d -> b h d x y', b=b, h=h, y=y)
        if self.residual_mode:
            for h in range(0, out.shape[1]):
                out[:, h, :, :, :] = out[:, h, :, :, :] + residual

        # out = out.view(out.shape[0], out.shape[1], out.shape[2], -1).mean(dim=3)
        # return self.layerNorm(out)

        out_ = self.norm[0](out[:, 0, :, :, :]).unsqueeze(1)
        for h in range(1, out.shape[1]):
            out_ = torch.cat((out_, self.norm[h](out[:, h, :, :, :]).unsqueeze(1)), dim=1)
        return out_.view(out_.shape[0], out_.shape[1], out_.shape[2], -1).mean(dim=3)


class AttentionMLP(nn.Module):
    def __init__(self, in_dim, head_dim=64, heads=16, residual_mode=False):
        super().__init__()
        self.h = heads
        self.scale = head_dim ** -0.5
        self.inner_dim = head_dim * self.h
        self.patch_size = 1
        self.q = nn.Linear(in_dim, self.inner_dim, bias=False)
        self.k = nn.Linear(in_dim, self.inner_dim, bias=False)
        self.v = nn.Linear(in_dim, self.inner_dim, bias=False)
        self.residual_mode = residual_mode
        if self.residual_mode:
            if head_dim != in_dim:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_dim, head_dim, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(head_dim),
                )
            else:
                self.downsample = nn.Identity()
        self.to_latent = nn.Identity()
        self.norm = nn.LayerNorm(head_dim)
        self.inp_norm = nn.LayerNorm(in_dim)
        self.out_norm = nn.LayerNorm(head_dim)

    def forward(self, x):
        if self.residual_mode:
            residual = self.downsample(x)
            residual = rearrange(residual, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                                 p1=self.patch_size, p2=self.patch_size)
            residual = residual.mean(dim=1)
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        q, k, v = self.q(x), self.k(x), self.v(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.h)

        qk = torch.einsum('bhid,bhjd->bhij', q, k)
        p_att = (qk * self.scale).softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', p_att, v)

        out = out.view(out.shape[0], out.shape[1], out.shape[2], -1).mean(dim=2)

        if self.residual_mode:
            for h in range(out.shape[1]):
                out[:, h, :] = out[:, h, :] + residual
        return self.out_norm(out)


class SeqAttention(nn.Module):
    def __init__(self, in_dim, head_dim, n_heads, sqa_type, residual_mode=False):
        super().__init__()
        if sqa_type == 'linear':
            self.sqa = AttentionMLP(in_dim, head_dim, n_heads, residual_mode)
        elif sqa_type == 'convolution':
            self.sqa = AttentionCNN(in_dim, head_dim, n_heads, residual_mode)

    def forward(self, x):
        return self.sqa(x)


class GAU(nn.Module):
    def __init__(self, dim, query_key_dim=128, expansion_factor=2., add_residual=True, dropout=0., ):
        super().__init__()
        hidden_dim = int(expansion_factor * dim)

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.to_hidden = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),
            nn.SiLU()
        )

        self.to_qk = nn.Sequential(
            nn.Linear(dim, query_key_dim),
            nn.SiLU()
        )

        self.gamma = nn.Parameter(torch.ones(2, query_key_dim))
        self.beta = nn.Parameter(torch.zeros(2, query_key_dim))
        nn.init.normal_(self.gamma, std=0.02)

        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

        self.add_residual = add_residual

    def forward(self, x):
        seq_len = x.shape[-2]

        normed_x = self.norm(x)  # (bs,seq_len,dim)
        v, gate = self.to_hidden(normed_x).chunk(2, dim=-1)  # (bs,seq_len,seq_len)

        Z = self.to_qk(normed_x)  # (bs,seq_len,query_key_dim)

        QK = einsum('... d, h d -> ... h d', Z, self.gamma) + self.beta
        q, k = QK.unbind(dim=-2)

        sim = einsum('b i d, b j d -> b i j', q, k) / seq_len

        A = F.relu(sim) ** 2
        A = self.dropout(A)

        V = einsum('b i j, b j d -> b i d', A, v)
        V = V * gate

        out = self.to_out(V)

        if self.add_residual:
            out = out + x

        return out


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        # print(attn)

        output = torch.bmm(attn, v)
        return output, attn, log_attn


class MultiHeadAttention(nn.Module):  # for 64 channel
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, do_activation=True):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.do_activation = do_activation

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # activation here
        if self.do_activation:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))

        # activation here
        if self.do_activation:
            output = self.activation(output)

        output = self.layer_norm(output + residual)

        return output


class TransformerEncoder(nn.Module):
    '''
    https://arxiv.org/pdf/2010.11929.pdf
    The Transformer encoder (Vaswani et al., 2017) consists of alternating layers of multiheaded self-attention (MSA, see Appendix A) and MLP blocks (Eq. 2, 3).
    Layernorm (LN) is applied before every block, and residual connections after every block (Wang et al., 2019; Baevski & Auli, 2019).
    The MLP contains two layers with a GELU non-linearity.
    '''

    def __init__(self, layers, hidden_size, mlp_size, heads):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(
                # nn.Sequential内部实现了forward函数，因此可以不用写forward函数。而nn.ModuleList则没有实现内部forward函数。
                nn.ModuleList([

                    # L and N 层
                    nn.LayerNorm((hidden_size)),  # norm

                    # 多头注意力层
                    nn.MultiheadAttention(  # multi-head attention
                        embed_dim=hidden_size,
                        num_heads=heads,
                        bias=False,  # 不要bias
                    ),
                    # nn.Sequential里面的模块按照顺序进行排列的，所以必须确保前一个模块的输出大小和下一个模块的输入大小是一致的。

                    # 线性层
                    nn.Sequential(  # mlp
                        nn.Linear(hidden_size, mlp_size),
                        nn.GELU(),
                        nn.Linear(mlp_size, hidden_size),
                    )
                ])
            )

    def forward(self, x):
        for layer in self.layers:
            # msa 这里应该代表 multi-head attention
            layer_norm, msa, mlp = layer
            # z_l′= MSA(LN(z_{l-1})) +z_{l−1}
            normed = layer_norm(x)
            x = msa(normed, normed, normed)[0] + x
            # z_l= MLP(LN(z_l')) +z_l'
            x = mlp(layer_norm(x)) + x
        return x


import math


class RelativePosition2d(nn.Module):
    '''
        Input size: (N, E, H, W)
        output size: (N, E, H, W)
    '''

    def __init__(self, d_model, max_size):
        super().__init__()
        # dimension will be divided into [x, y]
        assert d_model % 2 == 0
        d_emb = d_model // 2
        max_len = max(max_size)
        # Compute the positional encodings once in log space.
        # 计算位置编码

        pe = torch.zeros(max_len, d_emb)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_emb, 2) * -(math.log(10000.0) / d_emb))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        emb_h = self.pe[:x.size(2), :].unsqueeze(0).unsqueeze(0).permute(0, 3, 2, 1)
        # 比如图片img的size比如是（28，28，3）就可以利用img.permute(2,0,1)得到一个size为（3，28，28）的tensor
        emb_w = self.pe[:x.size(3), :].unsqueeze(0).unsqueeze(0).permute(0, 3, 1, 2)
        x[:, :x.size(1) // 2, :, :] += emb_h
        x[:, x.size(1) // 2:, :, :] += emb_w
        return x


class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError(
                "If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


class MlpBlock(nn.Module):

    def __init__(self, hidden_dim, mlp_dim):
        super().__init__()
        self.mlp_dim = mlp_dim
        self.dense_1 = nn.Linear(hidden_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.dense_2 = nn.Linear(mlp_dim, hidden_dim)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.gelu(x)
        x = self.dense_2(x)
        return x


class MixerBlock(nn.Module):
    def __init__(self, hidden_dim, token_dim, token_mlp_dim, channel_mlp_dim):
        super().__init__()
        self.mlp_token = MlpBlock(token_dim, token_mlp_dim)
        self.mlp_channel = MlpBlock(hidden_dim, channel_mlp_dim)
        self.layer_norm_1 = nn.LayerNorm(hidden_dim)
        self.layer_norm_2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        y = self.layer_norm_1(x)  ####### B,N,C

        # y = y.permute(0,1, 3, 2) 
        y = y.permute(0, 2, 1)  #### B,C,N

        y = self.mlp_token(y)

        # y = y.permute(0,1, 3, 2)
        y = y.permute(0, 2, 1)  #### B,N,C

        x = x + y
        y = self.layer_norm_2(x)
        return x + self.mlp_channel(y)


class GRUModel(nn.Module):

    def __init__(self, input_num, hidden_num, output_num):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_num
        # 这里设置了 batch_first=True, 所以应该 inputs = inputs.view(inputs.shape[0], -1, inputs.shape[1])
        # 针对时间序列预测问题，相当于将时间步（seq_len）设置为 1。
        self.GRU_layer = nn.GRU(input_size=input_num, hidden_size=hidden_num, num_layers=3, batch_first=True,
                                bidirectional=True)
        self.output_linear = nn.Linear(hidden_num * 2, output_num)
        self.hidden = None

    def forward(self, x):
        # h_n of shape (num_layers * num_directions, batch, hidden_size)
        # 这里不用显式地传入隐层状态 self.hidden
        x, self.hidden = self.GRU_layer(x)
        x = self.output_linear(x)
        return x, self.hidden


class ProtoNet(FewShotModel):
    def __init__(self, args):
        super().__init__(args)
        self.temp = nn.Parameter(torch.tensor(10., requires_grad=True))
        self.method = 'cos'
        # self.temp = torch.tensor(0.005,requires_grad=True)#nn.Parameter(torch.tensor(10.,requires_grad=True))
        # self.method = 'sqr'
        self.args = args
        self.laten_dim = 640
        self.slf_attn = MultiHeadAttention(1, self.laten_dim, self.laten_dim, self.laten_dim, dropout=0.2)
        self.slf_attn3 = MultiHeadAttention(1, self.laten_dim, self.laten_dim, self.laten_dim, dropout=0.2)
        self.transformer_encoder = TransformerEncoder(3, self.laten_dim, self.laten_dim, 2)
        self.GAU2 = GAU(self.laten_dim, self.laten_dim, dropout=0.2)
        self.conv1 = nn.Sequential(
            nn.Conv1d(self.laten_dim, self.laten_dim, kernel_size=2, bias=True),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv1d(self.laten_dim, self.laten_dim, kernel_size=2, bias=True),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv1d(self.laten_dim, self.laten_dim, kernel_size=2, bias=True),
            nn.ReLU(inplace=True))
        self.fc_0 = nn.Sequential(
            nn.Linear(self.laten_dim * 2, self.laten_dim, bias=True),
            nn.ReLU(inplace=True))

        self.num_grid = args.num_grid
        self.num_edge = args.num_edge
        self.num_patch = args.num_patch
        self.position_encoder = RelativePosition2d(d_model=self.laten_dim, max_size=(self.num_edge, self.num_edge))
        self.dropout = nn.Dropout(0.1)
        self.GRU = GRUModel(self.laten_dim, self.laten_dim, self.laten_dim)

    def compute_logits(self, feat, proto, metric='dot', temp=1.0):
        assert feat.dim() == proto.dim()
        if feat.dim() == 2:
            if metric == 'dot':
                logits = torch.mm(feat, proto.t())
            elif metric == 'cos':
                logits = 1 - torch.mm(F.normalize(feat, dim=-1),
                                      F.normalize(proto, dim=-1).t())  #### 1- 是t-s-s加上的
            elif metric == 'sqr':
                logits = -(feat.unsqueeze(1) -
                           proto.unsqueeze(0)).pow(2).sum(dim=-1)

        elif feat.dim() == 3:
            if metric == 'dot':
                logits = torch.bmm(feat, proto.permute(0, 2, 1))
            elif metric == 'cos':
                logits = torch.bmm(F.normalize(feat, dim=-1),
                                   F.normalize(proto, dim=-1).permute(0, 2, 1))
            elif metric == 'sqr':
                logits = -(feat.unsqueeze(2) -
                           proto.unsqueeze(1)).pow(2).sum(dim=-1)
        return logits * temp

    def make_nk_label(self, n, k, ep_per_batch=1):
        label = torch.arange(n).unsqueeze(1).expand(n, k).reshape(-1)
        label = label.repeat(ep_per_batch)
        return label

    def global_local_calibrate(self, x_total_nolocal):  # 80, 5, self.laten_dim
        x_global = x_total_nolocal[:, 0, :]
        global_feature = x_total_nolocal[:, 0, :].unsqueeze(1)
        grid_feature = x_total_nolocal[:, 1:, :]
        grid_feature = grid_feature.permute(0, 2, 1).view(self.args.way * (self.args.shot + self.args.query),
                                                          self.laten_dim, self.num_edge, self.num_edge)
        grid_feature = grid_feature + self.position_encoder(grid_feature)
        grid_feature = grid_feature.view(self.args.way * (self.args.shot + self.args.query), self.laten_dim,
                                         -1).permute(0, 2, 1)
        grid_feature = self.slf_attn3(grid_feature, grid_feature, grid_feature)
        x_total_nolocal = torch.cat([global_feature.repeat(1, self.num_grid, 1), grid_feature], dim=1)
        x_total_nolocal, _ = self.GRU(x_total_nolocal)
        global_feature_pos = x_total_nolocal[:, :self.num_grid, :]  # .unsqueeze(1)
        grid_feature = grid_feature + x_total_nolocal[:, self.num_grid:, :]
        fuse_grid_feature = torch.cat([global_feature_pos, grid_feature], dim=-1)
        fuse_grid = self.fc_0(fuse_grid_feature)
        x_global = x_global + grid_feature.mean(1)
        x_total_nolocal = torch.cat([x_global.unsqueeze(1), fuse_grid], dim=1)

        return x_total_nolocal

    def local_global_loss_cos(self, x_total):

        x_gloabl, x_locals = x_total.permute(1, 0, 2)[0], x_total.permute(1, 0, 2)[1:]
        loss = torch.tensor(0)

        x_gloabl = x_gloabl.unsqueeze(0).repeat(self.num_grid, 1, 1)

        loss = torch.bmm(F.normalize(x_locals, dim=-1),
                         F.normalize(x_gloabl, dim=-1).permute(0, 2, 1))
        loss = loss.mean(0).mean(0).mean(0)
        return loss

    def get_info_nce_loss(self, x_total):
        x_gloabl, x_locals = x_total.permute(1, 0, 2)[0], x_total.permute(1, 0, 2)[1:]
        loss = torch.tensor(0)
        for ii in range(x_locals.shape[0]):
            # loss = loss + self.emd_loss(x_gloabl,x_locals[ii])
            loss = loss + self.InfoNCE_loss(x_gloabl, x_locals[ii])

        return loss

    def get_task_feature(self, x_shot_all):
        # x_shot_raw : 1,5,1,10,self.laten_dim
        x_shot_raw = x_shot_all.view(1, -1, self.laten_dim)
        task_feature = nn.Parameter(torch.rand(1, 1, self.laten_dim)).cuda()
        total = torch.cat((task_feature, x_shot_raw), dim=1)
        total = self.transformer_encoder(total)
        task_feature = total[:, 0, :]

        return task_feature  # 1,self.laten_dim

    def _forward(self, x_shot, x_query):
        # print(x_shot.shape)   1,5, 1, 10, self.laten_dim,
        # print(x_query.shape)  1, 75, 10, self.laten_dim
        x_shot_rand = x_shot[:, :, :, (1 + self.num_grid):, :]
        x_shot_nolocal = x_shot[:, :, :, 0:(1 + self.num_grid), :]

        x_query_rand = x_query[:, :, (1 + self.num_grid):, :]
        x_query_nolocal = x_query[:, :, 0:(1 + self.num_grid), :]

        task_feature_shot = self.get_task_feature(x_shot)
        task_feature = task_feature_shot
        task_queryset = x_query.view(1, self.args.way, -1, self.laten_dim)
        task_feature_query = self.get_task_feature(task_queryset)

        x_shot_nolocal = x_shot_nolocal.view(self.args.shot * self.args.way, -1, self.laten_dim)
        x_query_nolocal = x_query_nolocal.view(self.args.query * self.args.way, -1, self.laten_dim)
        x_total_nolocal = torch.cat([x_shot_nolocal, x_query_nolocal], dim=0)

        x_shot_all = x_shot[:, :, :, 0, :] + x_shot[:, :, :, 1:(1 + self.num_grid), :].mean(-2) + x_shot[:, :, :,
                                                                                                  (1 + self.num_grid):,
                                                                                                  :].mean(-2)
        x_query_all = x_query[:, :, 0, :] + x_query[:, :, 1: self.num_grid, :].mean(-2) + x_query[:, :,
                                                                                          (1 + self.num_grid):, :].mean(
            -2)
        x_shot_all = x_shot_all.mean(2)

        x_total_nolocal = self.global_local_calibrate(x_total_nolocal)

        global_to_local_loss = self.local_global_loss_cos(x_total_nolocal)

        x_shot_nolocal = x_total_nolocal[0:self.args.way * self.args.shot, :, :]  # 5,5,self.laten_dim
        x_query_nolocal = x_total_nolocal[self.args.way * self.args.shot:, :, :]  # 75,5,self.laten_dim

        x_shot_global = x_shot_nolocal[:, 0, :]  #
        x_shot_grid = x_shot_nolocal[:, 1:(1 + self.num_grid), :]  # 5,4,self.laten_dim

        task_proto_global = x_shot_global.mean(0).unsqueeze(0).unsqueeze(-1)
        task_proto_grid = x_shot_grid.mean(0).mean(0).unsqueeze(0).unsqueeze(-1)

        x_shot_global = x_shot_global.contiguous().view(self.args.way, -1, self.laten_dim).mean(1)
        x_shot_grid = x_shot_grid.contiguous().view(self.args.way, -1, self.laten_dim).mean(1)

        x_shot_rand = x_shot_rand.squeeze(0).contiguous().view(self.args.way, -1,
                                                               self.laten_dim)  ##.permute(1, 0, 2)   #  5, 5*5, self.laten_dim
        x_query_rand = x_query_rand.contiguous().squeeze(0)  # .contiguous()

        x_shot_rand = self.GAU2(x_shot_rand)
        x_query_rand = self.GAU2(x_query_rand)
        x_shot_rand = x_shot_rand.mean(1)  # 5,self.laten_dim             类原型
        x_query_rand = x_query_rand.mean(1)  # 75,self.laten_dim            类原型
        task_proto_rand = x_shot_rand.mean(0, True).unsqueeze(-1)  ## 1,self.laten_dim,1
        task_feature = task_feature.unsqueeze(-1)
        if self.training:
            task_query = task_feature_query.unsqueeze(-1)
            task_feature = torch.cat([task_feature, task_query], dim=0).mean(0, True)

        sim_rand = torch.cat([task_feature, task_proto_rand], dim=-1)
        sim_rand = self.conv1(sim_rand)
        sim_rand = sim_rand.mean(-1).mean(-1).mean(-1)

        sim_grid = torch.cat([task_feature, task_proto_grid], dim=-1)
        sim_grid = self.conv2(sim_grid)
        sim_grid = sim_grid.mean(-1).mean(-1).mean(-1)

        sim_global = torch.cat([task_feature, task_proto_global], dim=-1)
        sim_global = self.conv3(sim_global)
        sim_global = sim_global.mean(-1).mean(-1).mean(-1)

        x_shot_nolocal = x_shot_nolocal.view(self.args.way, self.args.shot, -1, self.laten_dim)

        x_support_all = torch.cat([x_shot_nolocal[:, :, 0, :].mean(1, True),
                                   x_shot_nolocal[:, :, 1:(1 + self.num_grid), :].mean(2).mean(1, True),
                                   x_shot_rand.unsqueeze(1)], dim=1).mean(1)
        x_queryset_all = torch.cat(
            [x_query_nolocal[:, 0, :].unsqueeze(1), x_query_nolocal[:, 1:(1 + self.num_grid), :].mean(1, True),
             x_query_rand.unsqueeze(1)], dim=1).mean(1)

        x_support_all = x_support_all.unsqueeze(0)
        x_queryset_all = x_queryset_all.unsqueeze(0)

        x_shot_all = x_shot_global * sim_global + x_shot_grid * sim_grid + x_shot_rand * sim_rand
        x_shot_all = (x_shot_all.unsqueeze(0) + x_support_all) / 2

        x_query_global = x_query_nolocal[:, 0, :]  #
        x_query_grid = x_query_nolocal[:, 1:(1 + self.num_grid), :]  # 5,4,self.laten_dim
        x_query_grid = x_query_grid.mean(1)

        x_query_all = x_query_global * sim_global + x_query_grid * sim_grid + x_query_rand * sim_rand
        x_query_all = (x_query_all.unsqueeze(0) + x_queryset_all) / 2

        if self.method == 'cos':
            x_shot_all = F.normalize(x_shot_all, dim=-1)
            x_query_all = F.normalize(x_query_all, dim=-1)
            metric = 'dot'
        elif self.method == 'sqr':
            # x_shot_all = x_shot_all.mean(dim=-2)
            metric = 'sqr'

        logits = self.compute_logits(
            x_query_all, x_shot_all, metric=metric, temp=self.temp)

        logits = logits.view(-1, 5)

        if self.training:

            task_loss = 1 - torch.mm(F.normalize(task_feature_query, dim=-1),
                                     F.normalize(task_feature_shot, dim=-1).t()).mean(0).mean(0)

            return logits, task_loss + global_to_local_loss * 0.01  # +
        else:
            return logits
