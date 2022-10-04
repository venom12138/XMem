"""
This file defines XMem, the highest level nn.Module interface
During training, it is used by trainer.py
During evaluation, it is used by inference_core.py

It further depends on modules.py which gives more detailed implementations of sub-modules
"""

import torch
import torch.nn as nn
from copy import deepcopy
from model.aggregate import aggregate
from model.modules import *
from model.memory_util import *
# from aggregate import aggregate
# from modules import *
# from memory_util import *
from torchsummary import summary
import clip
class XMem(nn.Module):
    def __init__(self, config, model_path=None, map_location=None):
        """
        model_path/map_location are used in evaluation only
        map_location is for converting models saved in cuda to cpu
        """
        super().__init__()
        model_weights = self.init_hyperparameters(config, model_path, map_location)

        self.single_object = config.get('single_object', False)
        print(f'Single object mode: {self.single_object}')

        self.key_encoder = KeyEncoder() # R50 前三个stage: 
        self.value_encoder = ValueEncoder(self.value_dim, self.hidden_dim, self.single_object)
        self.flow_encoder = FlowEncoder()
        self.flow_value_fuser = FlowValueFuser(self.value_dim, 256, self.value_dim)
        
        # clip model
        # clip_model,_ = clip.load("ViT-L/14@336px")
        # self.token_embedding = clip_model.token_embedding
        # self.positional_embedding = clip_model.positional_embedding
        # self.transformer = clip_model.transformer
        # self.ln_final= clip_model.ln_final
        # self.text_projection = clip_model.text_projection
        # del clip_model
        
        # Projection from f16 feature space to key/value space
        # indim:1024,即f16；outdim:key_dim
        self.key_proj = KeyProjection(1024, self.key_dim)

        self.decoder = Decoder(self.value_dim, self.hidden_dim)

        if model_weights is not None:
            self.load_weights(model_weights, init_as_zero_if_needed=True)

    def encode_key(self, frame, need_sk=True, need_ek=True): 
        # Determine input shape
        # frame:[B, num_frames, C, H, W]
        # num_frames is t
        if len(frame.shape) == 5:
            # shape is b*t*c*h*w
            # 需要reshape回去
            need_reshape = True
            b, t = frame.shape[:2]
            # flatten so that we can feed them into a 2D CNN
            # frame:[B, num_frames, C, H, W] -> [B*num_frames, C, H, W]
            frame = frame.flatten(start_dim=0, end_dim=1)
        elif len(frame.shape) == 4:
            # shape is b*c*h*w
            need_reshape = False
        else:
            raise NotImplementedError
        # f16:[B*num_frames, 1024, H//16, W//16]
        f16, f8, f4 = self.key_encoder(frame)
        # key:[B*num_frames, key_dim, H//16, W//16]
        # shrinkage:[B*num_frames, 1, H//16, W//16]
        # selection:[B*num_frames, key_dim, H//16, W//16]
        key, shrinkage, selection = self.key_proj(f16, need_sk, need_ek)
        # 需要reshape回去
        if need_reshape:
            # B*key_dim*T*H//16*W//16
            key = key.view(b, t, *key.shape[-3:]).transpose(1, 2).contiguous()
            if shrinkage is not None:
                shrinkage = shrinkage.view(b, t, *shrinkage.shape[-3:]).transpose(1, 2).contiguous()
            if selection is not None:
                selection = selection.view(b, t, *selection.shape[-3:]).transpose(1, 2).contiguous()

            # B*T*C*H*W
            f16 = f16.view(b, t, *f16.shape[-3:])
            f8 = f8.view(b, t, *f8.shape[-3:])
            f4 = f4.view(b, t, *f4.shape[-3:])

        return key, shrinkage, selection, f16, f8, f4

    def encode_value(self, frame, image_feat_f16, h16, masks, is_deep_update=True): 
        # masks:[B, max_obj_num, H, W]
        # num_objects is max_obj_num
        # masks就是first_frame_gt是one-hot的，所以others就是一大堆1，把其他的object给mask出来了
        num_objects = masks.shape[1]
        if num_objects != 1:
            # others就是除了第i个object之外的所有object sum起来
            # others:[B, max_obj_num, H, W]
            others = torch.cat([
                torch.sum(
                    masks[:, [j for j in range(num_objects) if i!=j]]
                , dim=1, keepdim=True)
            for i in range(num_objects)], 1)
        else:
            others = torch.zeros_like(masks)

        g16, h16 = self.value_encoder(frame, image_feat_f16, h16, masks, others, is_deep_update)

        return g16, h16
    
    # flow: B x num_frames x 2 x H x W
    # return: B x Cf x H/P x W/P; Cf=256
    def encode_flow(self, flow):
        return self.flow_encoder(flow)
    
    # memory_value: [B, max_obj_num, x_in_dim/value_dim, H//16, W//16]
    # flow: [b, f_in_dim, H//16, W//16]
    # return: b x max_obj_num x x_in_dim/value_dim x H/P x W/P
    def fuse_flow_value(self, mv, flow_feat):
        return self.flow_value_fuser(mv, flow_feat)
    
    # Used in training only. 
    # This step is replaced by MemoryManager in test time
    def read_memory(self, query_key, query_selection, memory_key, 
                    memory_shrinkage, memory_value):
        """
        query_key       : B * CK * H * W
        query_selection : B * CK * H * W
        memory_key      : B * CK * T * H * W
        memory_shrinkage: B * 1  * T * H * W
        memory_value    : B * num_objects * CV * T * H * W
        """
        # T就是num_ref_frames
        # num_objects就是max_obj_num
        batch_size, num_objects = memory_value.shape[:2]
        memory_value = memory_value.flatten(start_dim=1, end_dim=2) # B x num_objects x CV x T x H x W -> B x num_objects*CV x T x H x W
        # query selection是key_proj之后的结果，是一个[B, CK, H, W]的tensor
        # affinity 是 [B, THW//P, HW//p]的tensor(384/16*384/16=576)
        affinity = get_affinity(memory_key, memory_shrinkage, query_key, query_selection)
        # mv: B x num_objects*CV x T x H x W -> B x num_objects*CV x H x W
        memory = readout(affinity, memory_value)
        # memory: B x num_objects x CV x H x W
        memory = memory.view(batch_size, num_objects, self.value_dim, *memory.shape[-2:])

        return memory

    def segment(self, multi_scale_features, memory_readout,
                    hidden_state, selector=None, h_out=True, strip_bg=True): 
        # logits B x max_obj_num x 1 x H x W
        # hidden_state B x max_obj_num x 64 x H//16 x W//16
        hidden_state, logits = self.decoder(*multi_scale_features, hidden_state, memory_readout, h_out=h_out)
        prob = torch.sigmoid(logits) # B x max_obj_num x H x W
        if selector is not None:
            prob = prob * selector # selector是[B, max_obj_num, 1, 1]的tensor
            
        logits, prob = aggregate(prob, dim=1, return_logits=True)
        if strip_bg:
            # Strip away the background
            prob = prob[:, 1:]
        # logits: B x max_obj_num+1 x H x W
        return hidden_state, logits, prob

    def forward(self, mode, *args, **kwargs):
        if mode == 'encode_key':
            return self.encode_key(*args, **kwargs)
        elif mode == 'encode_value':
            return self.encode_value(*args, **kwargs)
        elif mode == 'read_memory':
            return self.read_memory(*args, **kwargs)
        elif mode == 'segment':
            return self.segment(*args, **kwargs)
        elif mode == 'encode_flow':
            return self.encode_flow(*args, **kwargs)
        elif mode == 'fuse_flow_value':
            return self.fuse_flow_value(*args, **kwargs)
        else:
            raise NotImplementedError

    def init_hyperparameters(self, config, model_path=None, map_location=None):
        """
        Init three hyperparameters: key_dim, value_dim, and hidden_dim
        If model_path is provided, we load these from the model weights
        The actual parameters are then updated to the config in-place

        Otherwise we load it either from the config or default
        """
        if model_path is not None:
            # load the model and key/value/hidden dimensions with some hacks
            # config is updated with the loaded parameters
            model_weights = torch.load(model_path, map_location=map_location)
            self.key_dim = model_weights['key_proj.key_proj.weight'].shape[0]
            self.value_dim = model_weights['value_encoder.fuser.block2.conv2.weight'].shape[0]
            self.disable_hidden = 'decoder.hidden_update.transform.weight' not in model_weights
            if self.disable_hidden:
                self.hidden_dim = 0
            else:
                self.hidden_dim = model_weights['decoder.hidden_update.transform.weight'].shape[0]//3
            print(f'Hyperparameters read from the model weights: '
                    f'C^k={self.key_dim}, C^v={self.value_dim}, C^h={self.hidden_dim}')
        else:
            model_weights = None
            # load dimensions from config or default
            if 'key_dim' not in config:
                self.key_dim = 64
                print(f'key_dim not found in config. Set to default {self.key_dim}')
            else:
                self.key_dim = config['key_dim']

            if 'value_dim' not in config:
                self.value_dim = 512
                print(f'value_dim not found in config. Set to default {self.value_dim}')
            else:
                self.value_dim = config['value_dim']

            if 'hidden_dim' not in config:
                self.hidden_dim = 64
                print(f'hidden_dim not found in config. Set to default {self.hidden_dim}')
            else:
                self.hidden_dim = config['hidden_dim']

            self.disable_hidden = (self.hidden_dim <= 0)

        config['key_dim'] = self.key_dim
        config['value_dim'] = self.value_dim
        config['hidden_dim'] = self.hidden_dim

        return model_weights

    def load_weights(self, src_dict, init_as_zero_if_needed=False):
        # Maps SO weight (without other_mask) to MO weight (with other_mask)
        load_strict = False
        for k in list(src_dict.keys()):
            
            if ('flow_encoder' in k) or ('flow_value_fuser' in k):
                load_strict = True
            if k == 'value_encoder.conv1.weight':
                if src_dict[k].shape[1] == 4:
                    print('Converting weights from single object to multiple objects.')
                    pads = torch.zeros((64,1,7,7), device=src_dict[k].device)
                    if not init_as_zero_if_needed:
                        print('Randomly initialized padding.')
                        nn.init.orthogonal_(pads)
                    else:
                        print('Zero-initialized padding.')
                    src_dict[k] = torch.cat([src_dict[k], pads], 1)
        # print(load_strict)
        sd_before_load = deepcopy(self.state_dict())
        msg = self.load_state_dict(src_dict, strict=load_strict)
        # print(f'missing keys:{msg}')
        sd_after_load = deepcopy(self.state_dict())
        same_keys = [k for k in sd_before_load if torch.equal(sd_before_load[k], sd_after_load[k])]
        new_keys = []
        for key in same_keys:
            # print(key)
            # print('bn' in key)
            if key.startswith('key_encoder') or key.startswith('value_encoder'):
                if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key: 
                    continue
            new_keys.append(key)
        print('-------------------- Loaded weights --------------------')
        print(f'Weights unloaded:{new_keys}')
        print('----------------------------')
        if load_strict == False:
            assert len(new_keys) == len(self.flow_encoder.state_dict().keys()) + \
                                    len(self.flow_value_fuser.state_dict().keys())