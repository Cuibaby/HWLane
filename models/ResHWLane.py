from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.models as models
from timm.models.layers import trunc_normal_, DropPath
from models.registry import NET
from .resnet import ResNetWrapper 
from .decoder import BUSD, PlainDecoder 

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
class DWConv(nn.Module):
    def __init__(self, dim,  drop_path=0., layer_scale_init_value=1e-6):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
  

        
    def forward(self, x, h, w):
        b,n,c = x.shape
        x = x.transpose(1,2).view(b,c,h,w)
        
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1,2)

        return x

class Conv1x1(nn.Module):
    def __init__(self,inchannl, ratio):
        super(Conv1x1, self).__init__()
        self.conv1 = nn.Conv2d(inchannl, inchannl * ratio, kernel_size=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(inchannl * ratio, inchannl , kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inchannl * ratio, eps=1e-03)
        self.bn2 = nn.BatchNorm2d(inchannl, eps=1e-03)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        # self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.norm = nn.LayerNorm(in_features)
    def forward(self, x):
        x = x.permute(0,2,3,1)
        x = self.norm(x)
        x = self.fc1(x)
        # x = self.dwconv(x,h,w)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.permute(0,3,1,2)
        return x


class Attention(nn.Module):
    """
    LSA: self attention within a group
    """
    def __init__(self, dim, drop_path=0., num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.1, proj_drop=0., layer_scale_init_value = 1e-6):
        
        super(Attention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
       
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, flag):
        input = x
        B, C, H, W = x.shape
        if flag == 0:
            x = x.permute(0,3,2,1).reshape(-1, H, C)
            qkv = self.qkv(x).reshape(B, W, -1, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        else:
            x = x.permute(0,2,3,1).reshape(-1, W, C)
            qkv = self.qkv(x).reshape(B, H, -1, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        # B, hw, ws*ws, 3, n_head, head_dim -> 3, B, W, n_head, H, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, W, n_head, H, head_dim
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, W, n_head, H, H
        att = attn
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)  #attn @ v: 0: B, W, n_head, H, dim, 1:  B, H, n_head, W, dim,
        
        x = (attn @ v).transpose(2, 3).flatten(3) # 0: B, W, H, C, 1: B, H, W, C
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        if flag == 0:
            x = x.permute(0, 3, 2, 1) # (B, W, H, C) -> (B, C, H, W)
        else:
            x = x.permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)
        x = input + self.drop_path(x)
        return x, att

class WAttention(nn.Module):
    """
    LSA: self attention within a group
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.1, proj_drop=0.1, mlp_ratio = 4):
        
        super(WAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mlp = Conv1x1(dim,mlp_ratio)
        self.relu = nn.ReLU()

    def forward(self, x):
        B, C, H, W = x.shape
        input = x
        x = x.permute(0,2,3,1).reshape(-1, W, C)
        qkv = self.qkv(x).reshape(B, H, -1, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        # B, hw, ws*ws, 3, n_head, head_dim -> 3, B, H, n_head, W, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, H, n_head, W, head_dim
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, H, n_head, W, W
        att = attn
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)  # attn @ v-> B, H, n_head, W, dim -> (t(2,3)) B, H, W, n_head, head_dim ->reshape B, H, W, C
        x = (attn @ v).transpose(2, 3).reshape(B, H, W, C).permute(0, 3, 1, 2) 
       
        x = self.proj_drop(x)
        x = self.relu(input + self.mlp(x))
        return x, att, q

class HAttention(nn.Module):
    """
    GSA: using a  key to summarize the information for a group to be efficient.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.1, proj_drop=0.1, mlp_ratio = 4):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mlp = Conv1x1(dim,mlp_ratio)
        
        self.relu = nn.ReLU()
      
    def forward(self, x):
        B, C, H, W = x.shape
        input = x
        x = x.permute(0,3,2,1).reshape(-1, H, C) #[BW,H,C]
        qkv = self.qkv(x).reshape(B, W, -1, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        # B, hw, ws*ws, 3, n_head, head_dim -> 3, B, W, n_head, H, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2] #[B, W, n_head, H, head_dim]
        attn = (q @ k.transpose(-2, -1)) * self.scale #[B, W, n_head, H, H]
        att = attn
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(2, 3).reshape(B, W, H, C).permute(0, 3, 2, 1)  #[B, W, n_head, H, head_dim]->[B, C, H, W]
        
        x = self.proj_drop(x)
        x = self.relu(input + self.mlp(x))
        return x, att, q

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.1, attn_drop=0.1,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class GroupBlock(nn.Module):
    def __init__(self, dim, num_heads, channel, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop=0.1, attn_drop=0.1,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(GroupBlock, self).__init__()
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.attnh = HAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop)
        self.attnw = WAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop)
        self.mlp1 = Conv1x1(dim,mlp_ratio) #Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop) #
        self.mlp2 = Conv1x1(dim,mlp_ratio) #Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop) #
        self.relu = nn.ReLU()
       
    def forward(self, x):
        h_x, h_att = self.attnh(x)
        x = self.relu(x + h_x)
      
        x = self.relu(x + self.mlp1(x))
        
        w_x, w_att = self.attnw(x)
      
        x = self.relu(x + w_x)

        x = self.relu(x + self.mlp2(x))
       
        return x, h_att, w_att

class DetailHead(nn.Module):
    def __init__(self,cfg=None):
        super(DetailHead, self).__init__()
        self.cfg = cfg
        self.conv1x1 = nn.Conv2d(self.cfg.mfia.input_channel, cfg.num_classes, kernel_size=1, stride=1, bias=False)
    def forward(self, x):
      
        x = self.conv1x1(x) # 75.5
        x = F.interpolate(x,size=[self.cfg.img_height,  self.cfg.img_width],
                           mode='bilinear', align_corners=False)
        
        return x

class ExistHead(nn.Module):
    def __init__(self, cfg=None):
        super(ExistHead, self).__init__()
        self.cfg = cfg
        self.dropout = nn.Dropout2d(0.1)  
        self.fc = nn.Linear(self.cfg.mfia.input_channel, cfg.num_classes-1)

    def forward(self, x):
        x = self.dropout(x)
        x = torch.squeeze(x)
        x = self.fc(x)
        x = torch.sigmoid(x)

        return x

@NET.register_module
class ResHWLane(nn.Module):
    def __init__(self, cfg):
        super(ResHWLane, self).__init__()
        self.cfg = cfg
        self.backbone = ResNetWrapper(cfg)
        # for name, p in self.backbone.named_parameters():
        #     p.requires_grad = False
        self.hblocks = nn.ModuleList()
        self.wblocks = nn.ModuleList()
        for i in range(self.cfg.depth):
            self.hblocks.append(HAttention(dim = self.cfg.mfia.input_channel, num_heads = 8))
            self.wblocks.append(WAttention(dim = self.cfg.mfia.input_channel, num_heads = 8))
            #  self.blocks.append(GroupBlock(dim = self.cfg.mfia.input_channel, num_heads = 8, channel = self.cfg.mfia.input_channel))
        self.decoder = eval(cfg.decoder)(cfg)
        self.heads = ExistHead(cfg) 
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))

    def forward(self, batch):
        fea = self.backbone(batch)
        h_att = []
        w_att = []
        hq = []
        wq = []
        feat = []
        feat.append(fea)
        if self.cfg.hw_type == 0:
            for i in range(self.cfg.depth):
                fea, h, q = self.hblocks[i](fea)
                h_att.append(h)
                hq.append(q)
                feat.append(fea)
            for i in range(self.cfg.depth):
                fea, w, q = self.wblocks[i](fea)
                w_att.append(w)
                wq.append(q)
                feat.append(fea)
        else:
            for i in range(self.cfg.depth):
                fea, h, q = self.hblocks[i](fea)
                h_att.append(h)
                hq.append(q)
                fea, w, q = self.wblocks[i](fea)
                w_att.append(w)
                wq.append(q)
                feat.append(fea)
        avg_fea = self.global_avg_pool(fea) 
        seg = self.decoder(fea)
        exist = self.heads(avg_fea)

        output = {'seg': seg, 'exist': exist, 'feat': feat, 'h_att': h_att, 'w_att': w_att, 'hq': hq, 'wq': wq}

        return output

