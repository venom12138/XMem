"""
modules.py - This file stores the rather boring network blocks.

x - usually means features that only depends on the image
g - usually means features that also depends on the mask. 
    They might have an extra "group" or "num_objects" dimension, hence
    batch_size * num_objects * num_channels * H * W

The trailing number of a variable usually denote the stride

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.group_modules import *
from model import resnet
from model.cbam import CBAM
# from group_modules import *
# import resnet
# from cbam import CBAM
from torchsummary import summary
from einops import rearrange

class FeatureFusionBlock(nn.Module):
    def __init__(self, x_in_dim, g_in_dim, g_mid_dim, g_out_dim):
        super().__init__()

        self.distributor = MainToGroupDistributor()
        self.block1 = GroupResBlock(x_in_dim+g_in_dim, g_mid_dim)
        self.attention = CBAM(g_mid_dim)
        self.block2 = GroupResBlock(g_mid_dim, g_out_dim)

    def forward(self, x, g):
        batch_size, num_objects = g.shape[:2]
        # 将x和g cat起来
        g = self.distributor(x, g)
        # distributor后 g:[b, max_obj_num, x_in_dim+g_in_dim, H//16, W//16]
        g = self.block1(g)
        # 经过block1之后，g:[b, max_obj_num, g_mid_dim, H//16, W//16]
        r = self.attention(g.flatten(start_dim=0, end_dim=1))
        r = r.view(batch_size, num_objects, *r.shape[1:])

        g = self.block2(g+r)
        # fuse之后 [b, max_obj_num, g_out_dim, H//16, W//16]
        return g

class ValueFuser(nn.Module):
    def __init__(self, x_in_dim, f_in_dim, t_in_dim, out_dim):
        super().__init__()

        self.block1 = GroupResBlock(x_in_dim+f_in_dim+t_in_dim, out_dim)
        self.attention = CBAM(out_dim)
        self.block2 = GroupResBlock(out_dim, out_dim)
    
    # memory_value: [b, max_obj_num, x_in_dim or value_dim, H//16, W//16]
    # flow_feat_16: [b, f_in_dim, H//16, W//16]
    # text: [b, t_in_dim]
    # return: [b, max_obj_num, out_dim/value_dim, H//16, W//16]
    def forward(self, memory_value, flow_feat_16=None, text_feat=None):
        batch_size, num_objects = memory_value.shape[:2]
        HbyP, WbyP = memory_value.shape[-2:]
        if flow_feat_16 != None:
            flow_feat_16 = flow_feat_16.unsqueeze(1).repeat(1, num_objects, 1, 1, 1) # B x max_obj_num x f_in_dim x H//16 x W//16
            memory_value = torch.cat([memory_value, flow_feat_16], dim=2) # B x max_obj_num x (x_in_dim+f_in_dim) x H//16 x W//16
        if text_feat != None:
            text_feat = text_feat.unsqueeze(1).unsqueeze(3).unsqueeze(3).repeat(1, num_objects, 1, HbyP, WbyP) # B x max_obj_num x t_in_dim x H//16 x W//16
            memory_value = torch.cat([memory_value, text_feat], dim=2) # B x max_obj_num x (x_in_dim+f_in_dim+t_in_dim) x H//16 x W//16        

        # distributor后 g:[b, max_obj_num, x_in_dim+f_in_dim, H//16, W//16]
        memory_value = self.block1(memory_value)
        # 经过block1之后，g:[b, max_obj_num, out_dim, H//16, W//16]
        r = self.attention(memory_value.flatten(start_dim=0, end_dim=1))
        r = r.view(batch_size, num_objects, *r.shape[1:])

        memory_value = self.block2(memory_value+r)
        
        # fuse之后 [b, max_obj_num, out_dim, H//16, W//16]
        return memory_value

# not used for text feature fusion
class CrossAttentionValueFuser(nn.Module):
    def __init__(self, x_in_dim, f_in_dim, hidden_channels=256):
        super().__init__()
        self.query_dimensionality_reduction = nn.Conv2d(
            f_in_dim, hidden_channels, kernel_size=1, stride=1, padding=0, bias=True
        )
        
        self.reference_dimensionality_reduction = nn.Conv2d(
            x_in_dim, hidden_channels, kernel_size=1, stride=1, padding=0, bias=True
        )
    # reference: memory_value: [b, max_obj_num, x_in_dim or value_dim, H//16, W//16]
    # query: flow_feat_16: [b, f_in_dim, H//16, W//16]
    # return: [b, max_obj_num, out_dim or value_dim, H//16, W//16]
    # query: flow; reference: memory_value
    def forward(self, query_features, reference_features):
        batch_size, num_objects = reference_features.shape[:2]
        HbyP, WbyP = reference_features.shape[-2:]
        if flow_feat_16 != None:
            flow_feat_16 = flow_feat_16.unsqueeze(1).repeat(1, num_objects, 1, 1, 1) # B x max_obj_num x f_in_dim x H//16 x W//16
        
        Q = self.query_dimensionality_reduction(query_features)
        K = self.reference_dimensionality_reduction(reference_features)
        V = rearrange(reference_features, "b c h w -> b c (h w)")
        attention_map = torch.einsum("bcij,bckl->bijkl", Q, K)
        attention_map = rearrange(attention_map, "b h1 w1 h2 w2 -> b h1 w1 (h2 w2)")
        attention_map = nn.Softmax(dim=3)(attention_map)
        attended_features = torch.einsum("bijp,bcp->bcij", attention_map, V)
        return attended_features

class HiddenUpdater(nn.Module):
    # Used in the decoder, multi-scale feature + GRU
    def __init__(self, g_dims, mid_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.g16_conv = GConv2D(g_dims[0], mid_dim, kernel_size=1)
        self.g8_conv = GConv2D(g_dims[1], mid_dim, kernel_size=1)
        self.g4_conv = GConv2D(g_dims[2], mid_dim, kernel_size=1)

        self.transform = GConv2D(mid_dim+hidden_dim, hidden_dim*3, kernel_size=3, padding=1)

        nn.init.xavier_normal_(self.transform.weight)

    def forward(self, g, h):
        g = self.g16_conv(g[0]) + self.g8_conv(downsample_groups(g[1], ratio=1/2)) + \
            self.g4_conv(downsample_groups(g[2], ratio=1/4))

        g = torch.cat([g, h], 2)

        # defined slightly differently than standard GRU, 
        # namely the new value is generated before the forget gate.
        # might provide better gradient but frankly it was initially just an 
        # implementation error that I never bothered fixing
        values = self.transform(g)
        forget_gate = torch.sigmoid(values[:,:,:self.hidden_dim])
        update_gate = torch.sigmoid(values[:,:,self.hidden_dim:self.hidden_dim*2])
        new_value = torch.tanh(values[:,:,self.hidden_dim*2:])
        new_h = forget_gate*h*(1-update_gate) + update_gate*new_value

        return new_h


class HiddenReinforcer(nn.Module):
    # Used in the value encoder, a single GRU
    def __init__(self, g_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.transform = GConv2D(g_dim+hidden_dim, hidden_dim*3, kernel_size=3, padding=1)

        nn.init.xavier_normal_(self.transform.weight)

    def forward(self, g, h):
        g = torch.cat([g, h], 2)

        # defined slightly differently than standard GRU, 
        # namely the new value is generated before the forget gate.
        # might provide better gradient but frankly it was initially just an 
        # implementation error that I never bothered fixing
        values = self.transform(g)
        forget_gate = torch.sigmoid(values[:,:,:self.hidden_dim])
        update_gate = torch.sigmoid(values[:,:,self.hidden_dim:self.hidden_dim*2])
        new_value = torch.tanh(values[:,:,self.hidden_dim*2:])
        new_h = forget_gate*h*(1-update_gate) + update_gate*new_value

        return new_h


class ValueEncoder(nn.Module):
    def __init__(self, value_dim, hidden_dim, single_object=False):
        super().__init__()
        
        self.single_object = single_object
        network = resnet.resnet18(pretrained=True, extra_dim=1 if single_object else 2)
        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu  # 1/2, 64
        self.maxpool = network.maxpool

        self.layer1 = network.layer1 # 1/4, 64
        self.layer2 = network.layer2 # 1/8, 128
        self.layer3 = network.layer3 # 1/16, 256

        self.distributor = MainToGroupDistributor()
        self.fuser = FeatureFusionBlock(1024, 256, value_dim, value_dim)
        if hidden_dim > 0:
            self.hidden_reinforce = HiddenReinforcer(value_dim, hidden_dim)
        else:
            self.hidden_reinforce = None

    def forward(self, image, image_feat_f16, h, masks, others, is_deep_update=True):
        # image_feat_f16 is the feature from the key encoder
        if not self.single_object:
            # g: [b,max_num_objects,2,h,w]
            # stack之后，g[i,j]就是第j个object和除了第j个object之外的其他object的mask cat在一起了
            g = torch.stack([masks, others], 2)
        else:
            g = masks.unsqueeze(2)
        # 将g和image cat起来了
        # g: [b, max_num_objects,3+2,h,w]
        g = self.distributor(image, g)

        batch_size, num_objects = g.shape[:2]
        # flatten之后, g: [b*max_num_objects, 3, h, w]
        g = g.flatten(start_dim=0, end_dim=1)

        g = self.conv1(g)
        g = self.bn1(g) # 1/2, 64
        g = self.maxpool(g)  # 1/4, 64
        g = self.relu(g) 

        g = self.layer1(g) # 1/4,64
        g = self.layer2(g) # 1/8,128
        g = self.layer3(g) # 1/16,256

        g = g.view(batch_size, num_objects, *g.shape[1:])
        # fuser之前，
        # g: [b, max_num_objects, 256, H//16, W//16]
        # image_feat_f16: [b, 1024, H//16, W//16]
        g = self.fuser(image_feat_f16, g)
        # fuse之后，
        # g: [b, max_num_objects, value_dim, H//16, W//16]
        if is_deep_update and self.hidden_reinforce is not None:
            # GRU
            h = self.hidden_reinforce(g, h)

        return g, h

class KeyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        network = resnet.resnet50(pretrained=True)
        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu  # 1/2, 64
        self.maxpool = network.maxpool

        self.res2 = network.layer1 # 1/4, 256
        self.layer2 = network.layer2 # 1/8, 512
        self.layer3 = network.layer3 # 1/16, 1024

    def forward(self, f):
        x = self.conv1(f) 
        x = self.bn1(x)
        x = self.relu(x)   # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        f4 = self.res2(x)   # 1/4, 256
        f8 = self.layer2(f4) # 1/8, 512
        f16 = self.layer3(f8) # 1/16, 1024

        return f16, f8, f4

class FlowEncoder(nn.Module):
    def __init__(self, ):
        super(FlowEncoder, self).__init__()
        network = resnet.resnet18(pretrained=True)
        # flow的channels=2，其余的block使用预训练权重 
        self.conv1 = nn.Conv2d(10, 64, kernel_size=7, stride=2, padding=3, bias=False) 
        self.bn1 = network.bn1
        self.relu = network.relu  # 1/2, 64
        self.maxpool = network.maxpool

        self.layer1 = network.layer1 # 1/4, 64
        self.layer2 = network.layer2 # 1/8, 128
        self.layer3 = network.layer3 # 1/16, 256
    
    # flow:B x num_frames x 2 x H x W
    def forward(self, flow):
        if len(flow.shape) == 5:
            batch_size, num_frames = flow.shape[:2]
            # flatten之后, flow: [b*num_frames, 2, H//16, W//16]
            flow = flow.flatten(start_dim=0, end_dim=1) # 
            need_reshape = True
        else:
            need_reshape = False 
        flow = self.conv1(flow)
        flow = self.bn1(flow)
        flow = self.relu(flow) 
        flow = self.maxpool(flow)  # 1/4, 64

        flow_4 = self.layer1(flow) # 1/4, 64
        flow_8 = self.layer2(flow_4) # 1/8, 128
        flow_16 = self.layer3(flow_8) # 1/16, 256
        if need_reshape:
            flow_16 = flow_16.view(batch_size, num_frames, *flow_16.shape[1:])
        
        return flow_16

class UpsampleBlock(nn.Module):
    def __init__(self, skip_dim, g_up_dim, g_out_dim, scale_factor=2):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_dim, g_up_dim, kernel_size=3, padding=1)
        self.distributor = MainToGroupDistributor(method='add')
        self.out_conv = GroupResBlock(g_up_dim, g_out_dim)
        self.scale_factor = float(scale_factor)

    def forward(self, skip_f, up_g):
        skip_f = self.skip_conv(skip_f) #  B, 512, H//8, W//8 - > B, 512, H//8, W//8
        g = upsample_groups(up_g, ratio=self.scale_factor) # B, max_num_objects, 512, H//16, W//16 -> B, max_num_objects, 512, H//8, W//8
        g = self.distributor(skip_f, g) # cat起来
        g = self.out_conv(g) # B，max_num_objects, 256, H//8, W//8
        return g


class KeyProjection(nn.Module):
    def __init__(self, in_dim, keydim):
        super().__init__()

        self.key_proj = nn.Conv2d(in_dim, keydim, kernel_size=3, padding=1)
        # shrinkage
        self.d_proj = nn.Conv2d(in_dim, 1, kernel_size=3, padding=1)
        # selection
        self.e_proj = nn.Conv2d(in_dim, keydim, kernel_size=3, padding=1)

        nn.init.orthogonal_(self.key_proj.weight.data)
        nn.init.zeros_(self.key_proj.bias.data)
    
    def forward(self, x, need_s, need_e):
        shrinkage = self.d_proj(x)**2 + 1 if (need_s) else None
        selection = torch.sigmoid(self.e_proj(x)) if (need_e) else None

        return self.key_proj(x), shrinkage, selection


class Decoder(nn.Module):
    def __init__(self, val_dim, hidden_dim):
        super().__init__()

        self.fuser = FeatureFusionBlock(1024, val_dim+hidden_dim, 512, 512)
        if hidden_dim > 0:
            self.hidden_update = HiddenUpdater([512, 256, 256+1], 256, hidden_dim)
        else:
            self.hidden_update = None
        
        self.up_16_8 = UpsampleBlock(512, 512, 256) # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(256, 256, 256) # 1/8 -> 1/4

        self.pred = nn.Conv2d(256, 1, kernel_size=3, padding=1, stride=1)

    def forward(self, f16, f8, f4, hidden_state, memory_readout, h_out=True):
        batch_size, num_objects = memory_readout.shape[:2] # max_obj_num

        if self.hidden_update is not None:
            g16 = self.fuser(f16, torch.cat([memory_readout, hidden_state], 2))
        else:
            g16 = self.fuser(f16, memory_readout)

        g8 = self.up_16_8(f8, g16) # 将f8和g16 upsample fuse一下 
        g4 = self.up_8_4(f4, g8) # 将f4和g8 upsample fuse一下, g4: B x max_obj_num x 256 x H//4 x W//4
        logits = self.pred(F.relu(g4.flatten(start_dim=0, end_dim=1))) # 

        if h_out and self.hidden_update is not None:
            g4 = torch.cat([g4, logits.view(batch_size, num_objects, 1, *logits.shape[-2:])], 2)
            hidden_state = self.hidden_update([g16, g8, g4], hidden_state)
        else:
            hidden_state = None
        
        logits = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=False)
        logits = logits.view(batch_size, num_objects, *logits.shape[-2:])
        # logits B x max_obj_num x 1 x H x W
        # hidden_state B x max_obj_num x 64 x H//16 x W//16
        return hidden_state, logits
