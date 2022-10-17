import torch
import torch.nn as nn
from einops import rearrange

class CoAttentionModule(nn.Module):
    def __init__(self, input_channels=2048, hidden_channels=256, attention_type="coam"):
        super().__init__()
        self.attention_layer = CrossAttentionValueFuser(input_channels, hidden_channels)

    def forward(self, left_features, right_features):
        weighted_r = self.attention_layer(left_features, right_features)
        weighted_l = self.attention_layer(right_features, left_features)
        left_attended_features = rearrange(
            [left_features, weighted_r], "two b c h w -> b (two c) h w"
        )                               
        right_attended_features = rearrange(
            [right_features, weighted_l], "two b c h w -> b (two c) h w"
        )
        return left_attended_features, right_attended_features
    
class CrossAttentionValueFuser(nn.Module):
    def __init__(self, x_in_dim, f_in_dim, hidden_channels=256):
        super().__init__()
        self.query_dimensionality_reduction = nn.Conv2d(
            x_in_dim, hidden_channels, kernel_size=1, stride=1, padding=0, bias=True
        )
        
        self.reference_dimensionality_reduction = nn.Conv2d(
            f_in_dim, hidden_channels, kernel_size=1, stride=1, padding=0, bias=True
        )
        
    def forward(self, query_features, reference_features):
        Q = self.query_dimensionality_reduction(query_features)
        K = self.reference_dimensionality_reduction(reference_features)
        V = rearrange(reference_features, "b c h w -> b c (h w)")
        attention_map = torch.einsum("bcij,bckl->bijkl", Q, K)
        attention_map = rearrange(attention_map, "b h1 w1 h2 w2 -> b h1 w1 (h2 w2)")
        attention_map = nn.Softmax(dim=3)(attention_map)
        attended_features = torch.einsum("bijp,bcp->bcij", attention_map, V)
        return attended_features

if __name__ == "__main__":
    mv = torch.randn(1, 4, 3, 16, 16)
    flow = torch.randn(1, 5, 16, 16)
    query_dimensionality_reduction = nn.Conv2d(
            5, 64, kernel_size=1, stride=1, padding=0, bias=True
        )
    reference_dimensionality_reduction = nn.Conv2d(
            3, 64, kernel_size=1, stride=1, padding=0, bias=True
        )
    # Q = query_dimensionality_reduction(mv)
    # K = reference_dimensionality_reduction(dummy_input2)
    # V = rearrange(dummy_input2, "b c h w -> b c (h w)")
    # print('V')
    # print(V.shape)
    # attention_map = torch.einsum("bcij,bckl->bijkl", Q, K)
    # print('attention_map1')
    # print(attention_map.shape)
    # attention_map = rearrange(attention_map, "b h1 w1 h2 w2 -> b h1 w1 (h2 w2)")
    # print('attention_map2')
    # print(attention_map.shape)
    # attention_map = nn.Softmax(dim=3)(attention_map)
    # print('attention_map3')
    # print(attention_map.shape)
    # attended_features = torch.einsum("bijp,bcp->bcij", attention_map, V)
    
    # print(attended_features.shape)
    batch_size, num_objects = mv.shape[:2]
    HbyP, WbyP = mv.shape[-2:]
    
    query_features = flow.unsqueeze(1).repeat(1, num_objects, 1, 1, 1).flatten(start_dim=0, end_dim=1) # B x max_obj_num x f_in_dim x H//16 x W//16
    reference_features = mv.flatten(start_dim=0, end_dim=1)
    print(query_features.shape)
    print(reference_features.shape)
    Q = query_dimensionality_reduction(query_features)
    K = reference_dimensionality_reduction(reference_features)
    V = rearrange(reference_features, "b c h w -> b c (h w)")
    attention_map = torch.einsum("bcij,bckl->bijkl", Q, K)
    attention_map = rearrange(attention_map, "b h1 w1 h2 w2 -> b h1 w1 (h2 w2)")
    attention_map = nn.Softmax(dim=3)(attention_map)
    attended_features = torch.einsum("bijp,bcp->bcij", attention_map, V)
    attended_features = attended_features.view(batch_size, num_objects, *attended_features.shape[1:])
    print(attended_features.shape)