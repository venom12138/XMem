import torch
import torch.nn as nn
from einops import rearrange

class CoAttentionModule(nn.Module):
    def __init__(self, input_channels=2048, hidden_channels=256, attention_type="coam"):
        super().__init__()
        self.attention_layer = CoAttentionLayer(input_channels, hidden_channels)

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
    dummy_input = torch.randn(1, 3, 16, 16)
    dummy_input2 = torch.randn(1, 5, 16, 16)
    query_dimensionality_reduction = nn.Conv2d(
            3, 64, kernel_size=1, stride=1, padding=0, bias=True
        )
    reference_dimensionality_reduction = nn.Conv2d(
            5, 64, kernel_size=1, stride=1, padding=0, bias=True
        )
    Q = query_dimensionality_reduction(dummy_input)
    K = reference_dimensionality_reduction(dummy_input2)
    V = rearrange(dummy_input2, "b c h w -> b c (h w)")
    print('V')
    print(V.shape)
    attention_map = torch.einsum("bcij,bckl->bijkl", Q, K)
    print('attention_map1')
    print(attention_map.shape)
    attention_map = rearrange(attention_map, "b h1 w1 h2 w2 -> b h1 w1 (h2 w2)")
    print('attention_map2')
    print(attention_map.shape)
    attention_map = nn.Softmax(dim=3)(attention_map)
    print('attention_map3')
    print(attention_map.shape)
    attended_features = torch.einsum("bijp,bcp->bcij", attention_map, V)
    
    print(attended_features.shape)