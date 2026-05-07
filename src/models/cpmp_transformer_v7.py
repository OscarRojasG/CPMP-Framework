import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base.transformer import Transformer

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
        
        self.pos_embedding = nn.Embedding(H_dim + 1, d_model) # +1 por el token CLS
        
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
            num_layers=num_layers
        )
        
        self.origin_proj = nn.Linear(d_model, d_model)
        self.dest_proj = nn.Linear(d_model, d_model)

    def encode(self, S, X, H, memory=None):
        """
        S: (batch_size, S_len, H, C_dim)
        X: (batch_size, S_len, X_dim)
        memory: dict (opcional) {tuple_state: embedding_tensor}
        """
        batch_size, S_len, H_max, C_dim = S.shape
        device = S.device
        
        if memory is None:
            memory = {}

        # 1. Identificar qué estados ya tenemos en caché
        # Creamos una representación hashable de cada estado en el batch
        # Nota: S[i] es el estado completo de todos los stacks para la muestra i
        state_keys = [tuple(s.detach().cpu().numpy().flatten()) for s in S]
        
        # Índices de los estados que NO están en memoria
        missing_indices = [i for i, key in enumerate(state_keys) if key not in memory]
        
        # Tensor final que vamos a rellenar
        stack_embeddings = torch.zeros((batch_size, S_len, self.d_model), device=device)

        # 2. Si hay estados nuevos, procesarlos por el modelo
        if len(missing_indices) > 0:
            S_to_process = S[missing_indices] # [B', S, H, C]
            X_to_process = X[missing_indices]
            curr_B = S_to_process.shape[0]

            # 1. Preparar Máscara de Padding (True donde hay -1)
            # S_to_process == -1 en todas sus features N
            padding_mask = (S_to_process == -1).all(dim=-1) # [B', S, H]
            
            # 2. Proyección y Reshape
            x = self.input_projection(S_to_process.float()) # [B', S, H, d_model]
            x = x.view(curr_B * S_len, H_max, self.d_model) # [B'*S, H, d_model]
            
            # 3. Añadir CLS Token al inicio de cada secuencia (pila)
            cls_tokens = self.cls_token.expand(curr_B * S_len, 1, -1) # [B'*S, 1, d_model]
            x = torch.cat((cls_tokens, x), dim=1) # [B'*S, H+1, d_model]

            # 4. Positional Encoding
            positions = torch.arange(H_max + 1, device=device).unsqueeze(0)
            x = x + self.pos_embedding(positions)

            # 5. Máscara de atención para el CLS y contenedores reales
            # El CLS nunca es padding (False). Los contenedores son padding si eran -1.
            cls_mask = torch.zeros((curr_B * S_len, 1), dtype=torch.bool, device=device)
            full_padding_mask = padding_mask.view(curr_B * S_len, H_max)
            full_padding_mask = torch.cat((cls_mask, full_padding_mask), dim=1) # [B'*S, H+1]

            # 6. Intra-stack Attention
            # src_key_padding_mask hace que los -1 no influyan en el softmax
            x_out = self.intra_stack_attention(x, src_key_padding_mask=full_padding_mask)

            # 7. Pooling: Tomamos solo el output de la posición del CLS (índice 0)
            stack_vertical_info = x_out[:, 0, :].view(curr_B, S_len, self.d_model)
            
            # 8. Fusion con X (información global del stack)
            x_external_info = self.x_projection(X_to_process)
            combined = torch.cat([stack_vertical_info, x_external_info], dim=-1)
            stack_embeddings = self.fusion_norm(self.fusion_layer(combined))

        # 4. Recuperar los que ya estaban en memoria
        existing_indices = [i for i, key in enumerate(state_keys) if key in memory and i not in missing_indices]
        for i in existing_indices:
            stack_embeddings[i] = memory[state_keys[i]].to(device)

        return stack_embeddings, memory

    def decode(self, S, X, H, stack_embeddings):
        batch_size, S_len, H_max, C_dim = S.shape
        device = S.device

        x_global = self.inter_stack_attention(stack_embeddings)
        
        q_origin = self.origin_proj(x_global)
        k_dest = self.dest_proj(x_global)
        
        logits_matrix = torch.matmul(q_origin, k_dest.transpose(-1, -2)) / (self.d_model**0.5)

        mask_diag = torch.eye(S_len, device=device).bool().unsqueeze(0)
        
        # 1. Origen vacío (Sigue igual)
        is_origin_empty = (S == -1).all(dim=-1).all(dim=2) # (batch, S_len)

        # 2. Destino lleno (Lógica para H como tensor)
        # H viene como [h1, h2, ...]. Queremos el índice H-1.
        # Ajustamos H para que tenga la forma adecuada para indexar: (B, 1, 1, 1)
        # y luego expandirlo a la estructura de S.
        h_idx = (H - 1).view(batch_size, 1, 1, 1).expand(-1, S_len, 1, C_dim)
        
        # Extraemos la fila correspondiente a la altura H de cada stack
        # target_height_slice tendrá forma (batch, S_len, 1, C_dim)
        target_height_slice = torch.gather(S, 2, h_idx)
        
        # Eliminamos la dimensión extra de la altura y verificamos si está lleno
        # is_dest_full tendrá forma (batch, S_len)
        is_dest_full = ~(target_height_slice.squeeze(2) == -1).all(dim=-1) 
        
        # 3. Máscaras para la matriz de logits
        mask_origin = is_origin_empty.unsqueeze(2).expand(-1, -1, S_len) # (B, S_len, S_len)
        mask_dest = is_dest_full.unsqueeze(1).expand(-1, S_len, -1)     # (B, S_len, S_len)
        
        invalid_action_mask = mask_diag | mask_origin | mask_dest
        
        logits_matrix = logits_matrix.masked_fill(invalid_action_mask, -1e4)
        
        # El resto del código para aplanar los logits se mantiene igual...
        indices = torch.arange(S_len, device=device)
        src_grid = indices.view(-1, 1).repeat(1, S_len)
        dst_grid = indices.view(1, -1).repeat(S_len, 1)
        mask_diag_flat = src_grid != dst_grid
        
        logits = logits_matrix[:, mask_diag_flat]

        return logits
    
    def forward(self, S, X, H):
        stack_embeddings, _ = self.encode(S, X, H)
        logits = self.decode(S, X, H, stack_embeddings)
        return logits