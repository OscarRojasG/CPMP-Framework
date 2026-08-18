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
        
        # ELIMINADO: self.attention_sink = nn.Parameter(torch.randn(1, 1, d_model))
        
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
        memory: dict (opcional) {tuple_stack: embedding_tensor_1D}
        """
        batch_size, S_len, H_max, C_dim = L.shape
        device = L.device

        # s_mask es [B, S_len], True para stacks válidos, False para padding.
        s_mask = torch.arange(S_len, device=device).expand(batch_size, S_len) < S.unsqueeze(1)
        
        if memory is None:
            memory = {}

        # 1. Aplanar L y X para aislar CADA PILA individualmente
        L_flat_tensor = L.view(batch_size * S_len, H_max, C_dim)
        X_flat_tensor = X.view(batch_size * S_len, self.X_dim)
        
        # OPTIMIZACIÓN CRÍTICA: Un solo traslado a CPU para todo el batch
        L_flat_list = L_flat_tensor.detach().cpu().view(batch_size * S_len, -1).tolist()
        X_flat_list = X_flat_tensor.detach().cpu().tolist()
        
        # Generar llaves por pila (combinamos L y X para asegurar unicidad)
        stack_keys = [tuple(l + x) for l, x in zip(L_flat_list, X_flat_list)]
        
        # Filtrar solo las pilas que NUNCA hemos procesado
        missing_indices = [i for i, key in enumerate(stack_keys) if key not in memory]
        
        # 2. Si hay pilas nuevas, las procesamos TODAS JUNTAS en un sub-batch
        if len(missing_indices) > 0:
            L_missing = L_flat_tensor[missing_indices] # [N_missing, H, C]
            X_missing = X_flat_tensor[missing_indices] # [N_missing, X_dim]
            N = L_missing.shape[0]

            # Preparar Máscara de Padding (True donde hay -1)
            padding_mask = (L_missing == -1).all(dim=-1) # [N, H]
            
            # Proyección (ya no necesitamos hacer reshapes complejos, todo es de tamaño N)
            x = self.input_projection(L_missing.float()) # [N, H, d_model]
            
            # Añadir CLS Token
            cls_tokens = self.cls_token.expand(N, 1, -1) # [N, 1, d_model]
            x = torch.cat((cls_tokens, x), dim=1) # [N, H+1, d_model]

            # Máscara de atención para el CLS y contenedores reales
            cls_mask = torch.zeros((N, 1), dtype=torch.bool, device=device)
            full_padding_mask = torch.cat((cls_mask, padding_mask), dim=1) # [N, H+1]

            # Intra-stack Attention (procesa solo las N pilas faltantes)
            x_out = self.intra_stack_attention(x, src_key_padding_mask=full_padding_mask)

            # Pooling: Tomamos el CLS
            stack_vertical_info = x_out[:, 0, :] # [N, d_model]
            
            # Fusion con X
            x_external_info = self.x_projection(X_missing) # [N, d_model]
            combined = torch.cat([stack_vertical_info, x_external_info], dim=-1)
            final_embeddings = self.fusion_norm(self.fusion_layer(combined)) # [N, d_model]
            
            # Guardar en memoria (guardamos tensores 1D sueltos en GPU)
            for i, original_idx in enumerate(missing_indices):
                memory[stack_keys[original_idx]] = final_embeddings[i].detach()

        # 3. Reconstruir el tensor batch recuperando todo desde la memoria
        flat_embeddings = torch.stack([memory[key] for key in stack_keys])
        
        # Volvemos a darle la forma de tu Batch original
        stack_embeddings = flat_embeddings.view(batch_size, S_len, self.d_model)
        
        # Aplicamos la máscara de padding a nivel de Layout para las pilas inexistentes
        current_s_mask = s_mask.unsqueeze(-1)
        stack_embeddings = (stack_embeddings * current_s_mask).to(torch.float32)

        return stack_embeddings, memory
    
    def decode(self, stack_embeddings, L, X, S, H):
        batch_size, S_len, H_max, C_dim = L.shape
        device = L.device
    
        # Máscara para el transformer y el pooling (True = es padding)
        inter_padding_mask = ~(torch.arange(S_len, device=device).expand(batch_size, S_len) < S.unsqueeze(1))
    
        # Transformer inter-stack
        z = self.inter_stack_attention(stack_embeddings, src_key_padding_mask=inter_padding_mask)
        
        # 1. Capa lineal para calcular logits (ahora solo sobre las pilas 'z')
        attn_logits = self.cost_attention(z)
        
        # 2. Aplicar máscara directamente (sin concatenar una falsa para el sumidero)
        attn_logits = attn_logits.masked_fill(inter_padding_mask.unsqueeze(-1), -1e9)
        
        # 3. Softmax: Ahora el 100% de la probabilidad SE REPARTE SÍ O SÍ entre las pilas válidas
        attn_weights = torch.softmax(attn_logits, dim=1)
        
        # 4. Suma ponderada con los pesos
        z_global = torch.sum(z * attn_weights, dim=1)
    
        return self.cost_head(z_global).squeeze(-1)