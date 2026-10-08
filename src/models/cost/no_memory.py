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
        
        self.attention_sink = nn.Parameter(torch.randn(1, 1, d_model))
        
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
        memory: Se mantiene por compatibilidad de firma, pero se ignora y retorna None.
        """
        batch_size, S_len, H_max, C_dim = L.shape
        device = L.device

        # s_mask es [B, S_len], True para stacks válidos, False para padding.
        s_mask = torch.arange(S_len, device=device).expand(batch_size, S_len) < S.unsqueeze(1)
        
        # 1. Aplanar L y X para procesar TODAS las pilas en paralelo sin caché
        L_flat = L.view(batch_size * S_len, H_max, C_dim)
        X_flat = X.view(batch_size * S_len, self.X_dim)
        N = L_flat.shape[0]

        # Preparar Máscara de Padding (True donde hay -1)
        padding_mask = (L_flat == -1).all(dim=-1) # [N, H]
        
        # Proyección de entrada
        x = self.input_projection(L_flat.float()) # [N, H, d_model]
        
        # Añadir CLS Token
        cls_tokens = self.cls_token.expand(N, 1, -1) # [N, 1, d_model]
        x = torch.cat((cls_tokens, x), dim=1) # [N, H+1, d_model]

        # Máscara de atención para el CLS y contenedores reales
        cls_mask = torch.zeros((N, 1), dtype=torch.bool, device=device)
        full_padding_mask = torch.cat((cls_mask, padding_mask), dim=1) # [N, H+1]

        # Intra-stack Attention (procesa el batch completo de pilas repetidas y nuevas)
        x_out = self.intra_stack_attention(x, src_key_padding_mask=full_padding_mask)

        # Pooling: Tomamos el CLS
        stack_vertical_info = x_out[:, 0, :] # [N, d_model]
        
        # Fusion con X
        x_external_info = self.x_projection(X_flat) # [N, d_model]
        combined = torch.cat([stack_vertical_info, x_external_info], dim=-1)
        final_embeddings = self.fusion_norm(self.fusion_layer(combined)) # [N, d_model]
        
        # 3. Volver a darle la forma de tu Batch original
        stack_embeddings = final_embeddings.view(batch_size, S_len, self.d_model)
        
        # Aplicamos la máscara de padding a nivel de Layout para las pilas inexistentes
        current_s_mask = s_mask.unsqueeze(-1)
        stack_embeddings = (stack_embeddings * current_s_mask).to(torch.float32)

        return stack_embeddings, None
    
    def decode(self, stack_embeddings, L, X, S, H):
        batch_size, S_len, H_max, C_dim = L.shape
        device = L.device
    
        # Máscara para el transformer y el pooling (True = es padding)
        inter_padding_mask = ~(torch.arange(S_len, device=device).expand(batch_size, S_len) < S.unsqueeze(1))
    
        # Pasa la máscara al TransformerEncoder para que los stacks válidos no atiendan a los ceros del padding
        z = self.inter_stack_attention(stack_embeddings, src_key_padding_mask=inter_padding_mask)
        
        # 1. Añadimos el vector sumidero a la secuencia procesada
        sink = self.attention_sink.expand(batch_size, 1, -1)
        z_with_sink = torch.cat([z, sink], dim=1) # [B, S_len + 1, d_model]
        
        # 2. Capa lineal para calcular logits (sobre todos + el sumidero)
        attn_logits = self.cost_attention(z_with_sink)
        
        # 3. Ajustar la máscara: El sumidero NUNCA es padding (False)
        sink_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
        full_mask = torch.cat([inter_padding_mask, sink_mask], dim=1) # [B, S_len + 1]
        
        # 4. Aplicar máscara.
        attn_logits = attn_logits.masked_fill(full_mask.unsqueeze(-1), -1e9)
        
        # 5. Softmax: Ahora, si ningún stack es importante, la red le da el peso al sumidero
        attn_weights = torch.softmax(attn_logits, dim=1)
        
        # 6. Suma ponderada con los pesos
        z_global = torch.sum(z_with_sink * attn_weights, dim=1)
    
        return self.cost_head(z_global).squeeze(-1)