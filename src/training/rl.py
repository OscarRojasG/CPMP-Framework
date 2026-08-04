import os
import copy
import json
import time
import random
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from cpmp.layout import Layout
from solvers.model import ModelSolver
from training.common import save_model, LRConfig

# =====================================================================
# CONFIGURACIÓN
# =====================================================================

@dataclass
class POMOConfig:
    updates: int = 10000            # Total de iteraciones de entrenamiento
    k_rollouts: int = 16            # Trayectorias independientes por instancia (K)
    instances_per_combo: int = 8    # Instancias frescas por cada combo en cada update
    kl_coef: float = 0.03           # Penalización de divergencia KL vs modelo base
    adv_clip: float = 4.0           # Límite máximo/mínimo para la ventaja
    grad_clip: float = 1.0          # Límite para la norma del gradiente
    minibatch_size: int = 2048      # Tamaño de batch en la fase de replay (forward plano)
    eval_interval: int = 100        # Frecuencia de evaluación en el dev-set
    patience: int = 15              # Evaluaciones sin mejora antes de parar

@torch.no_grad()
def sample_pomo_rollouts(
    model: torch.nn.Module, 
    instances: List[Layout], 
    k_rollouts: int, 
    max_steps_dict: Dict[Tuple[int, int], int], 
    input_adapter_config: Tuple[Any, ...], 
    device: torch.device
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, List[Tuple[int, int]], torch.Tensor, torch.Tensor, float]:
    """
    Juega K partidas en paralelo para cada instancia de la lista usando muestreo multinomial.
    Retorna los historiales y la entropía normalizada promedio del muestreo.
    """
    model.eval()
    la_class, *la_args = input_adapter_config
    
    total_envs = len(instances) * k_rollouts
    layouts = []
    combo_ids = []
    max_steps_array = []
    
    # 1. Instanciar los K rollouts
    for inst in instances: 
        combo = (len(inst.stacks), inst.H)
        max_s = max_steps_dict.get(combo, 150)
        for _ in range(k_rollouts):
            layouts.append(Layout([list(s) for s in inst.stacks], inst.H))
            combo_ids.append(combo)
            max_steps_array.append(max_s)
            
    active_mask = np.ones(total_envs, dtype=bool)
    steps_taken = np.zeros(total_envs, dtype=int)
    
    flat_inputs_steps = []
    flat_actions = []
    flat_row = []
    norm_entropy_history = [] 
    
    # 2. Bucle de simulación paralela
    while np.any(active_mask):
        active_indices = np.where(active_mask)[0]
        
        # --- A. Vectorización ---
        input_adapter = la_class(*la_args)
        for idx in active_indices:
            lay = layouts[idx]
            vec = input_adapter.input_2_vec(lay, lay.H) 
            input_adapter.add(vec)
            
        batch_dict = input_adapter.get()
        
        batch_inputs = []
        for key in batch_dict:
            tensor = torch.tensor(batch_dict[key], device=device)
            batch_inputs.append(tensor)
            
        # --- B. Inferencia de la Red ---
        stack_embeddings, _ = model.encode(*batch_inputs)
        logits = model.decode(stack_embeddings, *batch_inputs)
        
        # --- C. Enmascaramiento ---
        is_valid = logits > -100.0
        probs = torch.softmax(logits.float(), dim=-1)
        
        valid_counts = is_valid.sum(dim=-1).float()
        max_entropies = torch.log(torch.clamp(valid_counts, min=1.0))
        log_probs = torch.log(probs + 1e-8)
        step_entropies = -(probs * log_probs).sum(dim=-1)
        step_norm_entropies = step_entropies / torch.clamp(max_entropies, min=1e-8)
        norm_entropy_history.extend(step_norm_entropies.tolist())
        
        # --- D. Muestreo POMO ---
        sampled_actions = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        flat_inputs_steps.append([b.cpu() for b in batch_inputs])
        flat_actions.append(sampled_actions.cpu())
        flat_row.append(torch.tensor(active_indices, dtype=torch.long))
        
        # --- E. Aplicar Movimiento ---
        for i, idx in enumerate(active_indices):
            lay = layouts[idx]
            action_idx = sampled_actions[i].item()
            
            src = action_idx // (input_adapter.S_max - 1)
            remainder = action_idx % (input_adapter.S_max - 1)
            dst = remainder + 1 if remainder >= src else remainder
            
            lay.move(src, dst)
            steps_taken[idx] += 1
            
            if lay.is_sorted() or steps_taken[idx] >= max_steps_array[idx]:
                active_mask[idx] = False

    # 3. Asignación de Recompensas Finales
    returns = torch.zeros(total_envs, dtype=torch.float32, device=device)
    is_solved = torch.zeros(total_envs, dtype=torch.bool, device=device)
    for idx in range(total_envs):
        if layouts[idx].is_sorted():
            returns[idx] = -steps_taken[idx]
            is_solved[idx] = True
        else:
            returns[idx] = -max_steps_array[idx]
            is_solved[idx] = False
            
    states_history = [torch.cat(tensors, dim=0) for tensors in zip(*flat_inputs_steps)]
    actions_history = torch.cat(flat_actions)
    rollout_idx_history = torch.cat(flat_row)
    
    current_norm_ent = float(np.mean(norm_entropy_history)) if norm_entropy_history else 1.0
    
    return states_history, actions_history, returns, combo_ids, rollout_idx_history, is_solved, current_norm_ent

def compute_pomo_advantages(
    returns: torch.Tensor, 
    combo_ids: List[Tuple[int, int]], 
    rollout_idx_history: torch.Tensor,
    k_rollouts: int, 
    clip_val: float,
    device: torch.device
) -> torch.Tensor:
    """
    Calcula la ventaja POMO y la propaga al historial de pasos.
    """
    total_envs = returns.shape[0]
    num_instances = total_envs // k_rollouts
    
    # 1. Baseline por instancia (Auto-competencia POMO)
    returns_matrix = returns.view(num_instances, k_rollouts)
    baseline = returns_matrix.mean(dim=1, keepdim=True)
    
    # Ventaja bruta SIN clipping previo
    advantages = (returns_matrix - baseline).view(total_envs)
    
    # 2. Normalización por combo (S, H)
    combo_to_indices = {}
    for env_idx, combo in enumerate(combo_ids):
        if combo not in combo_to_indices:
            combo_to_indices[combo] = []
        combo_to_indices[combo].append(env_idx)
        
    advantages_norm = advantages.clone()
    eps = 1e-6
    
    for combo, indices in combo_to_indices.items():
        indices_tensor = torch.tensor(indices, dtype=torch.long, device=device)
        combo_adv = advantages[indices_tensor]
        std = combo_adv.std()
        
        if torch.isnan(std) or std < eps:
            advantages_norm[indices_tensor] = 0.0
        else:
            advantages_norm[indices_tensor] = combo_adv / (std + eps)
            
    # 3. Único Clipping Final como red de seguridad
    advantages_norm = torch.clamp(advantages_norm, min=-clip_val, max=clip_val)
            
    # 4. Propagación temporal
    advantages_history = advantages_norm[rollout_idx_history.to(device)]
    
    return advantages_history

def compute_pomo_loss(
    logits: torch.Tensor, 
    ref_logits: torch.Tensor, 
    actions: torch.Tensor, 
    advantages: torch.Tensor, 
    kl_coef: float
):
    """
    Calcula la función objetivo completa de la Fase 2:
    L = - (Adv * log π(a|s)) + kl_coef * KL(π || π_ref)
    """
    logits = logits.float()
    ref_logits = ref_logits.float()
    
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    
    with torch.no_grad():
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        
    # TÉRMINO 1: Policy Gradient 
    log_prob_actions = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    pg_loss = -(advantages * log_prob_actions).mean()
    
    # TÉRMINO 2: Entropía (Solo para monitoreo, ya no afecta la pérdida)
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    
    # TÉRMINO 3: Ancla KL
    kl_div = (probs * (log_probs - ref_log_probs)).sum(dim=-1).mean()
    
    # PÉRDIDA TOTAL
    total_loss = pg_loss + (kl_coef * kl_div)
    
    return total_loss, pg_loss.detach(), entropy.detach(), kl_div.detach()

def update_model_pomo(
    model: torch.nn.Module, 
    ref_model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    states_history: List[torch.Tensor], 
    actions_history: torch.Tensor, 
    advantages_history: torch.Tensor, 
    minibatch_size: int, 
    kl_coef: float, 
    grad_clip: float, 
    device: torch.device
):
    """
    Realiza el forward plano acumulando gradientes en chunks de minibatch_size.
    Aplica clip_grad_norm_ y da un solo paso de optimización.
    """
    model.eval()
    ref_model.eval()

    total_steps = actions_history.shape[0]
    num_inputs = len(states_history)
    
    total_pg_loss = 0.0
    total_entropy = 0.0
    total_kl = 0.0
    
    optimizer.zero_grad(set_to_none=True)
    indices = torch.randperm(total_steps)
    
    for start_idx in range(0, total_steps, minibatch_size):
        end_idx = min(start_idx + minibatch_size, total_steps)
        mb_indices = indices[start_idx:end_idx]
        mb_indices_dev = mb_indices.to(device)
        mb_size = end_idx - start_idx
        
        mb_states = [states_history[j][mb_indices].to(device) for j in range(num_inputs)]
        mb_actions = actions_history[mb_indices].to(device)
        mb_advantages = advantages_history[mb_indices_dev]
        
        stack_embeddings, _ = model.encode(*mb_states)
        logits = model.decode(stack_embeddings, *mb_states)
        
        with torch.no_grad():
            ref_embeddings, _ = ref_model.encode(*mb_states)
            ref_logits = ref_model.decode(ref_embeddings, *mb_states)
                
        loss, pg, ent, kl = compute_pomo_loss(
            logits, ref_logits, mb_actions, mb_advantages, kl_coef
        )
        
        ratio = mb_size / total_steps
        scaled_loss = loss * ratio
        
        scaled_loss.backward()
        
        total_pg_loss += pg.item() * ratio
        total_entropy += ent.item() * ratio
        total_kl += kl.item() * ratio
        
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    
    return total_pg_loss, total_entropy, total_kl

@torch.no_grad()
def evaluate_dev_greedy(
    model: torch.nn.Module, 
    dev_instances: List[Layout], 
    max_steps_dict: dict, 
    input_adapter_config: tuple,
    device: torch.device
) -> Tuple[float, int, int]:
    """
    Evalúa el modelo delegando la inferencia y validación de ciclos a ModelSolver.
    Agrupa las instancias por dimensiones (S, H) para evaluar eficientemente en batch.
    """
    model.eval()
    la_class, *la_args = input_adapter_config
    input_adapter = la_class(*la_args)
    
    total_envs = len(dev_instances)
    penalized_steps = []
    solved_count = 0
    
    # Agrupar las instancias de dev por (S, H) 
    grouped_instances = defaultdict(list)
    for lay in dev_instances:
        S = len(lay.stacks)
        H = lay.H
        grouped_instances[(S, H)].append(lay)
        
    for (S, H), lays in grouped_instances.items():
        lays_copy = [copy.deepcopy(lay) for lay in lays]
        max_steps = max_steps_dict.get((S, H), 150)
        
        solver = ModelSolver(model, input_adapter, batch_size=256)
        results = solver.solve_from_layouts(lays_copy, H, max_steps)
        
        for solved, steps in results:
            if solved:
                solved_count += 1
                penalized_steps.append(steps)
            else:
                penalized_steps.append(max_steps)

    return float(np.mean(penalized_steps)), solved_count, total_envs

def train_pomo_rl(
    model: torch.nn.Module, 
    ref_model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer,
    generators: list,    
    train_instances_dict: dict,
    dev_instances: list,       
    pomo_config: POMOConfig,               
    max_steps_dict: dict,
    input_adapter_config: tuple,               
    device: torch.device,
    save_dir: str,
    model_name: str
):
    print(f"Iniciando POMO RL por {pomo_config.updates} updates...")
    
    os.makedirs(save_dir, exist_ok=True)
    best_dev_score = float('inf')
    
    evals_without_improvement = 0
    history = []
    
    for update in range(1, pomo_config.updates + 1):
        start_time = time.time()
        
        instances = []
        for generator in generators:
            combo_instances = random.sample(train_instances_dict[generator], pomo_config.instances_per_combo)
            instances.extend(combo_instances)
                
        states_hist, actions_hist, returns, combo_ids, rollout_idx_hist, is_solved, current_norm_ent = sample_pomo_rollouts(
            model, instances, 
            pomo_config.k_rollouts, 
            max_steps_dict,  
            input_adapter_config,
            device
        )
        
        advantages_hist = compute_pomo_advantages(
            returns, combo_ids, rollout_idx_hist, 
            pomo_config.k_rollouts, 
            pomo_config.adv_clip, 
            device
        )
        
        pg_loss, entropy, kl_div = update_model_pomo(
            model, ref_model, optimizer,
            states_hist, actions_hist, advantages_hist,
            pomo_config.minibatch_size, pomo_config.kl_coef,
            pomo_config.grad_clip, device
        )
        
        # Telemetría 
        solved_rate = is_solved.float().mean().item() * 100
        avg_steps = -returns[is_solved].mean().item() if solved_rate > 0 else 0.0
        fps = len(actions_hist) / max(1e-4, time.time() - start_time)
        
        # Logging
        print(f"Update {update:05d} | "
              f"PG: {pg_loss:+.3f} | Ent: {entropy:.3f} | KL: {kl_div:.3f} | "
              f"Solve: {solved_rate:3.0f}% | AvgSteps: {avg_steps:.1f} | "
              f"NormEnt: {current_norm_ent:.2f} | {fps:.0f} steps/s")
              
        history.append({
            "update": update, "pg_loss": pg_loss, "entropy": entropy, 
            "kl_div": kl_div, "solve_rate": solved_rate,
            "norm_ent": current_norm_ent
        })
        
        # Evaluación en Dev-Set
        if update % pomo_config.eval_interval == 0:
            dev_score, solved_count, total_dev = evaluate_dev_greedy(
                model, dev_instances, max_steps_dict, 
                input_adapter_config, device
            )
            solve_rate_dev = (solved_count / total_dev) * 100
            
            print(f"   >>> 🧪 Evaluación Dev: {dev_score:.2f} pasos | Resueltas: {solved_count}/{total_dev} ({solve_rate_dev:.1f}%) | (Mejor: {min(dev_score, best_dev_score):.2f})")
            
            if dev_score < best_dev_score:
                best_dev_score = dev_score
                
                evals_without_improvement = 0
                
                save_model(model, model_name)
                print(f"   >>> 🏆 ¡Nuevo mejor modelo guardado!")
                ref_model.load_state_dict(model.state_dict())
                print(f"   >>> ⚓ Política de referencia (ref_model) actualizada.")
            else:
                evals_without_improvement += 1
                print(f"   >>> ⚠️ Sin mejora ({evals_without_improvement}/{pomo_config.patience}).")
                
                if evals_without_improvement > pomo_config.patience:
                    print(f"   >>> 🛑 Entrenamiento detenido: se alcanzó el límite de evaluaciones sin mejora ({pomo_config.patience}).")
                    break
            
    # Guardar historial al finalizar
    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f)
        
    print("✅ Entrenamiento POMO finalizado.")

def train(
    model: torch.nn.Module, 
    ref_model: torch.nn.Module, 
    generators: list,     
    pomo_config: POMOConfig,               
    input_adapter_config: tuple,
    learning_rate: float,
    weight_decay: float,         
    device: torch.device,
    save_dir: str,
    model_name: str,
    train_size_per_combo: int,
    test_size_per_combo: int,
    seed: int
):
    max_steps_dict = {}
    for gen in generators:
        N = gen.S * (gen.H - 2)
        if N <= 20: steps = 60
        elif N <= 40: steps = 100
        else: steps = 150
        max_steps_dict[(gen.S, gen.H)] = steps

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    dev_instances = []
    train_instances_dict = {}
    
    print("Generando pool estático de instancias de entrenamiento y validación...")
    for gen in generators:
        combo_devs = gen.generate_instances(test_size_per_combo)
        dev_instances.extend([Layout(inst, gen.H) for inst in combo_devs])
        
        combo_trains = gen.generate_instances(train_size_per_combo)
        train_instances_dict[gen] = [Layout(inst, gen.H) for inst in combo_trains]

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )

    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    train_pomo_rl(
        model=model,
        ref_model=ref_model,
        optimizer=optimizer,
        generators=generators,
        train_instances_dict=train_instances_dict,
        dev_instances=dev_instances,
        pomo_config=pomo_config,
        max_steps_dict=max_steps_dict,
        input_adapter_config=input_adapter_config,
        device=device,
        save_dir=save_dir,
        model_name=model_name
    )