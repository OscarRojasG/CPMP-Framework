import torch
import torch.nn as nn
from models.transformer import Transformer

class CostPredictorTransformer(Transformer):
    def __init__(self, H, C_dim, X_dim, d_model=64, nhead=8, num_layers=2, ff_dim_multiplier=4, dropout=0.1):
        super().__init__(H=H, C_dim=C_dim, X_dim=X_dim, d_model=d_model, 
                         nhead=nhead, num_layers=num_layers, 
                         ff_dim_multiplier=ff_dim_multiplier, dropout=dropout)
        
        self.d_model = d_model
        self.H = H
        
        # Procesamiento de Stacks (Independiente)
        self.input_proj = nn.Linear(C_dim, d_model)
        self.empty_embed = nn.Parameter(torch.randn(1, 1, 1, d_model))
        self.intra_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * ff_dim_multiplier, dropout, batch_first=True),
            num_layers=num_layers, enable_nested_tensor=False
        )
        self.summary_layer = nn.Linear(H * d_model, d_model)
        
        # Fusión
        self.x_proj = nn.Linear(X_dim, d_model)
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.fusion_norm = nn.LayerNorm(d_model)
        
        # Inter-stack y Cabeza de Costo
        self.inter_stack = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * ff_dim_multiplier, dropout, batch_first=True),
            num_layers=num_layers
        )
        self.cost_attention = nn.Linear(d_model, 1)
        self.cost_head = nn.Sequential(
            nn.Linear(d_model, d_model * ff_dim_multiplier),
            nn.GELU(),
            nn.Linear(d_model * ff_dim_multiplier, d_model),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )
    
    def encode(self, L, X, S, H, memory=None):
        batch_size, S_len, H_max, C_dim = L.shape
        device = L.device
        
        padding_mask = (L == -1).all(dim=-1) 
        x = self.input_proj(L.float())
        x = torch.where(padding_mask.unsqueeze(-1), self.empty_embed, x)
        
        x = x.view(batch_size * S_len, H_max, self.d_model) 
        x = self.intra_attn(x) 
        
        x = x.view(batch_size, S_len, H_max, self.d_model)
        x_flat = x.view(batch_size, S_len, H_max * self.d_model)
        stack_vertical_info = self.summary_layer(x_flat) 
        
        x_external_info = self.x_proj(X) 
        combined = torch.cat([stack_vertical_info, x_external_info], dim=-1)
        stack_embeddings = self.fusion(combined) 
        stack_embeddings = self.fusion_norm(stack_embeddings)

        return stack_embeddings, memory

    def decode(self, stack_embeddings, L, X, S, H):  
        z = self.inter_stack(stack_embeddings)
        
        # Global Pooling por Atención
        attn_weights = torch.softmax(self.cost_attention(z), dim=1)
        z_global = torch.sum(z * attn_weights, dim=1)
        
        return self.cost_head(z_global).squeeze(-1)