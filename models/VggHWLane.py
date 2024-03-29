import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.models as models
from models.registry import NET
from .resnet import ResNetWrapper 
from .decoder import BUSD, PlainDecoder 

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
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class HAttention(nn.Module):
    """
    LSA: self attention within a group
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        
        super(HAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
       
        self.proj_drop = nn.Dropout(proj_drop)
       

    def forward(self, x):
       
        B, C, H, W = x.shape
        x = x.permute(0,3,2,1).reshape(-1, H, C)
        qkv = self.qkv(x).reshape(B, W, -1, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        # B, hw, ws*ws, 3, n_head, head_dim -> 3, B, W, n_head, H, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, W, n_head, H, head_dim
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, W, n_head, H, H
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(
            attn)  # attn @ v-> B, W, n_head, H, dim -> (t(2,3)) B, W, H, n_head, head_dim ->reshape B, W, H, C
        x = (attn @ v).transpose(2, 3).reshape(B, W, H, C).permute(0, 3, 2, 1)
        
        x = self.proj_drop(x)
       
        return x, attn

class WAttention(nn.Module):
    """
    LSA: self attention within a group
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        
        super(WAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
       
        self.proj_drop = nn.Dropout(proj_drop)
       

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0,2,3,1).reshape(-1, W, C)
        qkv = self.qkv(x).reshape(B, H, -1, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        # B, hw, ws*ws, 3, n_head, head_dim -> 3, B, H, n_head, W, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, H, n_head, W, head_dim
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, H, n_head, W, W
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(
            attn)  # attn @ v-> B, H, n_head, W, dim -> (t(2,3)) B, H, W, n_head, head_dim ->reshape B, H, W, C
        x = (attn @ v).transpose(2, 3).reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        x = self.proj_drop(x)
        return x, attn

class Attention(nn.Module):
    """
    GSA: using a  key to summarize the information for a group to be efficient.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
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
class VggHWLane(nn.Module):
    def __init__(self, cfg):
        super(VggHWLane, self).__init__()
        self.cfg = cfg
        self.backbone = models.vgg16_bn(pretrained=True).features #512

        # ----------------- process backbone -----------------
        for i in [34, 37, 40]:
            conv = self.backbone._modules[str(i)]
            dilated_conv = nn.Conv2d(
                conv.in_channels, conv.out_channels, conv.kernel_size, stride=conv.stride,
                padding=tuple(p * 2 for p in conv.padding), dilation=2, bias=(conv.bias is not None)
            )
            dilated_conv.load_state_dict(conv.state_dict())
            self.backbone._modules[str(i)] = dilated_conv
       
        self.backbone._modules.pop('33')
        self.backbone._modules.pop('43')
        # ----------------- SCNN part -----------------
        self.layer1 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, self.cfg.mfia.input_channel, 1, bias=False),
            nn.BatchNorm2d(self.cfg.mfia.input_channel),
            nn.ReLU() 
        )

    #    self.backbone = ResNetWrapper(cfg)
    #    for name, p in self.backbone.named_parameters():
    #        p.requires_grad = False
        self.blocks = nn.ModuleList()
        for i in range(self.cfg.depth):
             self.blocks.append(GroupBlock(dim = self.cfg.mfia.input_channel, num_heads = 8, channel = self.cfg.mfia.input_channel)) #num_heads = 16 2022_02_14 before
        self.decoder = eval(cfg.decoder)(cfg)
        self.heads = ExistHead(cfg) 
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))
    
    def forward(self, batch):
        fea = self.backbone(batch)
        fea = self.layer1(fea)
        h_att = []
        w_att = []
    #    low_fea = self.low_fea(fea)
        for i in range(self.cfg.depth):
            fea, h, w = self.blocks[i](fea)
            h_att.append(h)
            w_att.append(w)
        avg_fea = self.global_avg_pool(fea) 
        seg = self.decoder(fea)
        exist = self.heads(avg_fea)
        output = {'seg': seg, 'exist': exist, 'h_att': h_att, 'w_att': w_att}

        return output
