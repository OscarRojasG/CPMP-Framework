import torch
import torch.nn as nn
from models.transformer import Transformer

class CPMPTransformer(Transformer):
    def __init__(self, H_dim, C_dim, X_dim, d_model=64, nhead=8, num_layers=2, ff_dim_multiplier=4, dropout=0.1):
        super().__init__(
            H=H_dim,
            C_dim=C_dim,
            X_dim=X_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            ff_dim_multiplier=ff_dim_multiplier,
            dropout=dropout
        )
        self.d_model = d_model
        self.H = H_dim
        self.X_dim = X_dim
        self.C_dim = C_dim
        
        self.input_projection = nn.Linear(C_dim, d_model)
        self.empty_embed = nn.Parameter(torch.randn(1, 1, 1, d_model))
        
        self.intra_stack_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * ff_dim_multiplier, dropout, batch_first=True),
            num_layers=num_layers,
            enable_nested_tensor=False
        )
        
        self.stack_summary_layer = nn.Linear(H_dim * d_model, d_model)

        self.x_projection = nn.Linear(X_dim, d_model)
        self.fusion_layer = nn.Linear(d_model * 2, d_model)
        self.fusion_norm = nn.LayerNorm(d_model)
        
        self.inter_stack_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * ff_dim_multiplier, dropout, batch_first=True),
            num_layers=num_layers,
            enable_nested_tensor=False
        )
        
        self.origin_proj = nn.Linear(d_model, d_model)
        self.dest_proj = nn.Linear(d_model, d_model)

    def encode(self, L, X, S, H, memory=None):
        batch_size, S_len, H_max, C_dim = L.shape
        device = L.device
        
        padding_mask = (L == -1).all(dim=-1) 
        x = self.input_projection(L.float())
        x = torch.where(padding_mask.unsqueeze(-1), self.empty_embed, x)
        
        x = x.view(batch_size * S_len, H_max, self.d_model) 
        x = self.intra_stack_attention(x) 
        
        x = x.view(batch_size, S_len, H_max, self.d_model)
        x_flat = x.view(batch_size, S_len, H_max * self.d_model)
        stack_vertical_info = self.stack_summary_layer(x_flat) 
        
        x_external_info = self.x_projection(X) 
        combined = torch.cat([stack_vertical_info, x_external_info], dim=-1)
        stack_embeddings = self.fusion_layer(combined) 
        stack_embeddings = self.fusion_norm(stack_embeddings)

        return stack_embeddings, memory

    def decode(self, stack_embeddings, L, X, S, H):
        batch_size, S_len, H_max, C_dim = L.shape
        device = L.device

        # 1. Máscara para la atención Inter-Stack
        # El TransformerEncoder de PyTorch usa src_key_padding_mask
        # donde True significa "ignorar este token"
        inter_padding_mask = ~(torch.arange(S_len, device=device).expand(batch_size, S_len) < S.unsqueeze(1))
        
        # x_global solo contendrá info de los primeros S stacks
        x_global = self.inter_stack_attention(stack_embeddings, src_key_padding_mask=inter_padding_mask)
        
        q_origin = self.origin_proj(x_global)
        k_dest = self.dest_proj(x_global)
        
        logits_matrix = torch.matmul(q_origin, k_dest.transpose(-1, -2)) / (self.d_model**0.5)

        # 2. Máscaras de validez de movimiento
        mask_diag = torch.eye(S_len, device=device).bool().unsqueeze(0)
        
        # NUEVA MÁSCARA: Bloquear cualquier stack fuera de S
        # Si el índice de origen >= S o destino >= S, movimiento inválido
        out_of_bounds = inter_padding_mask # [B, S_len] (True donde es inválido)
        mask_out_S_origin = out_of_bounds.unsqueeze(2).expand(-1, -1, S_len)
        mask_out_S_dest = out_of_bounds.unsqueeze(1).expand(-1, S_len, -1)

        # (Lógica anterior de vacío/lleno)
        is_origin_empty = (L == -1).all(dim=-1).all(dim=2) 
        h_idx = (H - 1).view(batch_size, 1, 1, 1).expand(-1, S_len, 1, C_dim)
        target_height_slice = torch.gather(L, 2, h_idx)
        is_dest_full = ~(target_height_slice.squeeze(2) == -1).all(dim=-1) 
        
        mask_origin = is_origin_empty.unsqueeze(2).expand(-1, -1, S_len)
        mask_dest = is_dest_full.unsqueeze(1).expand(-1, S_len, -1)
        
        # Unimos todas las restricciones
        invalid_action_mask = (
            mask_diag | 
            mask_origin | 
            mask_dest | 
            mask_out_S_origin |  # No puedes mover desde un stack que no existe
            mask_out_S_dest      # No puedes mover hacia un stack que no existe
        )
        
        logits_matrix = logits_matrix.masked_fill(invalid_action_mask, -1e4)

        # 3. Aplanar excluyendo la diagonal
        indices = torch.arange(S_len, device=device)
        src_grid = indices.view(-1, 1).repeat(1, S_len)
        dst_grid = indices.view(1, -1).repeat(S_len, 1)
        mask_no_diag = src_grid != dst_grid
        
        logits = logits_matrix[:, mask_no_diag].view(batch_size, -1)
        
        return logits