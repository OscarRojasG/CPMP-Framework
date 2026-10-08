import torch
import torch.nn as nn
from models.transformer import Transformer

class CPMPTransformer(Transformer):
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
        
        self.origin_proj = nn.Linear(d_model, d_model)
        self.dest_proj = nn.Linear(d_model, d_model)

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
        # torch.stack une todos los vectores 1D de la memoria en un bloque velozmente
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