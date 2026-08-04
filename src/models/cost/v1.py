import torch
import torch.nn as nn
from models.transformer import Transformer

class CostPredictorTransformer(Transformer):
    def __init__(self, H_dim, C_dim, X_dim, d_model=64, nhead=8, num_layers=2, ff_dim_multiplier=4, dropout=0.1):
        super().__init__(
            H_dim=H_dim,
            C_dim=C_dim,
            X_dim=X_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            ff_dim_multiplier=ff_dim_multiplier,
            dropout=dropout
        )
        self.d_model = d_model
        self.H_dim = H_dim
        self.X_dim = X_dim
        self.C_dim = C_dim
        
        self.input_projection = nn.Linear(C_dim, d_model)
        
        # Token CLS: Representará el resumen de la pila
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.intra_stack_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * ff_dim_multiplier, dropout, batch_first=True),
            num_layers=num_layers,
            enable_nested_tensor=False
        )

        self.x_projection = nn.Linear(X_dim, d_model)
        self.fusion_layer = nn.Linear(d_model * 2, d_model)
        self.fusion_norm = nn.LayerNorm(d_model)
        
        self.inter_stack_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * ff_dim_multiplier, dropout, batch_first=True),
            num_layers=num_layers,
            enable_nested_tensor=False
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
        """
        S: (batch_size, S_len, H, C_dim)
        X: (batch_size, S_len, X_dim)
        memory: dict (opcional) {tuple_state: embedding_tensor}
        """
        batch_size, S_len, H_max, C_dim = L.shape
        device = L.device

        # s_mask es [B, S_len], True para stacks válidos, False para padding.
        s_mask = torch.arange(S_len, device=device).expand(batch_size, S_len) < S.unsqueeze(1)

        # 1. Preparar Máscara de Padding (True donde hay -1)
        # L == -1 en todas sus features N
        padding_mask = (L == -1).all(dim=-1) # [B, S, H]
        
        # 2. Proyección y Reshape
        x = self.input_projection(L.float()) # [B, S, H, d_model]
        x = x.view(batch_size * S_len, H_max, self.d_model) # [B*S, H, d_model]
        
        # 3. Añadir CLS Token al inicio de cada secuencia (pila)
        cls_tokens = self.cls_token.expand(batch_size * S_len, 1, -1) # [B*S, 1, d_model]
        x = torch.cat((cls_tokens, x), dim=1) # [B*S, H+1, d_model]

        # 4. Máscara de atención para el CLS y contenedores reales
        # El CLS nunca es padding (False). Los contenedores son padding si eran -1.
        cls_mask = torch.zeros((batch_size * S_len, 1), dtype=torch.bool, device=device)
        full_padding_mask = padding_mask.view(batch_size * S_len, H_max)
        full_padding_mask = torch.cat((cls_mask, full_padding_mask), dim=1) # [B*S, H+1]

        # 5. Intra-stack Attention
        # src_key_padding_mask hace que los -1 no influyan en el softmax
        x_out = self.intra_stack_attention(x, src_key_padding_mask=full_padding_mask)

        # 6. Pooling: Tomamos solo el output de la posición del CLS (índice 0)
        stack_vertical_info = x_out[:, 0, :].view(batch_size, S_len, self.d_model)
        
        # 7. Fusion con X
        x_external_info = self.x_projection(X)
        combined = torch.cat([stack_vertical_info, x_external_info], dim=-1)
        processed = self.fusion_norm(self.fusion_layer(combined))
        
        # Aplicamos la máscara S a los embeddings recién calculados
        # Los stacks fuera de S se ponen en 0 (o un valor neutral)
        current_s_mask = s_mask.unsqueeze(-1)
        stack_embeddings = (processed * current_s_mask).to(torch.float32)

        return stack_embeddings, memory
    
    def decode(self, stack_embeddings, L, X, S, H):
        batch_size, S_len, H_max, C_dim = L.shape
        device = L.device
    
        # Zeroing de stacks de padding antes del encoder
        inter_padding_mask = ~(torch.arange(S_len, device=device).expand(batch_size, S_len) < S.unsqueeze(1))
        stack_embeddings = stack_embeddings * (~inter_padding_mask).unsqueeze(-1).float()
    
        # Inter-stack attention sin máscara
        z = self.inter_stack_attention(stack_embeddings)
    
        # Attention pooling con enmascaramiento final
        attn_logits = self.cost_attention(z)
        attn_logits = attn_logits.masked_fill(
            inter_padding_mask.unsqueeze(-1), -1e4
        )
        attn_weights = torch.softmax(attn_logits, dim=1)
        z_global = torch.sum(z * attn_weights, dim=1)
    
        return self.cost_head(z_global).squeeze(-1)