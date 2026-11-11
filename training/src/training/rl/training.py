import os
import copy
import json
import time
import random
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from xml.parsers.expat import model

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

from cpmp.layout import Layout
from generation.data import get_feasible_moves
from training.sl.training import save_model

# =====================================================================
# CONFIGURACIÓN
# =====================================================================

@dataclass
class POMOConfig:
    updates: int = 10000            # Total de iteraciones de entrenamiento
    k_rollouts: int = 16            # Trayectorias independientes por instancia (K)
    instances_per_combo: int = 8    # Instancias frescas por cada combo en cada update
    lr: float = 1e-5                # Learning rate constante
    ent_coef_start: float = 0.01    # Bono de entropía inicial
    ent_coef_end: float = 0.001     # Bono de entropía final (decaimiento lineal)
    kl_coef: float = 0.03           # Penalización de divergencia KL vs modelo base
    unsolved_extra: int = 10        # Castigo adicional si no se resuelve el layout
    adv_clip: float = 4.0           # Límite máximo/mínimo para la ventaja
    grad_clip: float = 1.0          # Límite para la norma del gradiente
    minibatch_size: int = 2048      # Tamaño de batch en la fase de replay (forward plano)
    eval_interval: int = 100        # Frecuencia de evaluación en el dev-set
    snapshot_interval: int = 1000   # Frecuencia de guardado de pesos

@torch.no_grad()
def sample_pomo_rollouts(
    model: torch.nn.Module,
    instances: List[Layout],
    k_rollouts: int,
    max_steps_dict: Dict[Tuple[int, int], int],
    input_adapter_config: Tuple[Any, ...],
    unsolved_extra: int,
    device: torch.device
) -> Tuple[List[List[torch.Tensor]], torch.Tensor, torch.Tensor, List[Tuple[int, int]], List[int], torch.Tensor]:
    """
    Juega K partidas en paralelo para cada instancia de la lista usando muestreo multinomial.
    Retorna:
      - states_history: Lista de listas de tensores (los inputs batcheados) por cada paso válido.
      - actions_history: Tensor 1D con los índices de las acciones tomadas en cada paso.
      - returns: Tensor 1D (tamaño N*K) con el retorno final de cada trayectoria.
      - combo_ids: Lista (tamaño N*K) identificando el combo (S, H) de cada trayectoria.
      - rollout_idx_history: Lista de enteros que mapea a qué trayectoria pertenece cada paso.
      - is_solved: Tensor booleano (tamaño N*K) indicando si la trayectoria resolvió el problema.
    """
    model.eval()
    la_class, *la_args = input_adapter_config

    total_envs = len(instances) * k_rollouts
    layouts = []
    combo_ids = []
    max_steps_array = []

    for inst in instances:
        combo = (len(inst.stacks), inst.H)
        max_s = max_steps_dict.get(combo, 150)
        for _ in range(k_rollouts):
            layouts.append(copy.deepcopy(inst))
            combo_ids.append(combo)
            max_steps_array.append(max_s)

    active_mask = np.ones(total_envs, dtype=bool)
    steps_taken = np.zeros(total_envs, dtype=int)
    
    # NUEVO: Memoria del último movimiento para evitar U-Turns inmediatos
    last_moves = [(-1, -1) for _ in range(total_envs)]

    states_history = []
    actions_history = []
    rollout_idx_history = []

    while np.any(active_mask):
        active_indices = np.where(active_mask)[0]

        input_adapter = la_class(*la_args)
        for idx in active_indices:
            lay = layouts[idx]
            vec = input_adapter.input_2_vec(lay, lay.H)
            input_adapter.add(vec)

        batch_dict = input_adapter.get()
        batch_inputs = [torch.tensor(batch_dict[key], device=device) for key in batch_dict]

        with autocast(device.type):
            logits = model(*batch_inputs)

        mask = torch.full_like(logits, -1e4)

        for i, idx in enumerate(active_indices):
            lay = layouts[idx]
            feasible_moves = get_feasible_moves(lay)

            for src, dst in feasible_moves:
                # NUEVO: Si el movimiento es exactamente devolver la pieza anterior, lo bloqueamos
                if (src, dst) == last_moves[idx]:
                    continue
                
                action_idx = src * (input_adapter.S_max - 1) + (dst - int(dst > src))
                mask[i, action_idx] = 0.0

        masked_logits = logits + mask
        probs = torch.softmax(masked_logits, dim=-1)

        sampled_actions = torch.multinomial(probs, num_samples=1).squeeze(-1)

        for i, idx in enumerate(active_indices):
            lay = layouts[idx]
            action_idx = sampled_actions[i].item()

            src = action_idx // (input_adapter.S_max - 1)
            remainder = action_idx % (input_adapter.S_max - 1)
            dst = remainder + 1 if remainder >= src else remainder

            lay.move(src, dst)
            steps_taken[idx] += 1
            
            # NUEVO: Guardamos el movimiento inverso (destino a origen) para el próximo turno
            last_moves[idx] = (dst, src)

            state_tensors = [b[i].clone() for b in batch_inputs]
            states_history.append(state_tensors)
            actions_history.append(action_idx)
            rollout_idx_history.append(idx)

            if lay.is_sorted() or steps_taken[idx] >= max_steps_array[idx]:
                active_mask[idx] = False

    returns = torch.zeros(total_envs, dtype=torch.float32, device=device)
    is_solved = torch.zeros(total_envs, dtype=torch.bool, device=device)
    for idx in range(total_envs):
        if layouts[idx].is_sorted():
            returns[idx] = -steps_taken[idx]
            is_solved[idx] = True
        else:
            returns[idx] = -(max_steps_array[idx] + unsolved_extra)
            is_solved[idx] = False

    actions_history = torch.tensor(actions_history, dtype=torch.long, device=device)

    return states_history, actions_history, returns, combo_ids, rollout_idx_history, is_solved

def compute_pomo_advantages(
    returns: torch.Tensor,
    combo_ids: List[Tuple[int, int]],
    rollout_idx_history: List[int],
    k_rollouts: int,
    clip_val: float,
    device: torch.device
) -> torch.Tensor:
    """
    Calcula la ventaja POMO y la propaga al historial de pasos.

    Args:
        returns: Tensor 1D de tamaño (N * K) con los retornos finales por entorno.
        combo_ids: Lista de tamaño (N * K) con las tuplas (S, H) de cada entorno.
        rollout_idx_history: Lista de índices que mapean cada paso a su entorno original.
        k_rollouts: Cantidad de rollouts por instancia.
        clip_val: Valor máximo/mínimo absoluto para la ventaja (ej. 4.0).
    """
    total_envs = returns.shape[0]
    num_instances = total_envs // k_rollouts

    # 1. Baseline por instancia (Auto-competencia POMO)
    # Redimensionamos a (N, K) para calcular el promedio de los K rollouts de cada instancia
    returns_matrix = returns.view(num_instances, k_rollouts)
    baseline = returns_matrix.mean(dim=1, keepdim=True)

    # Restamos el baseline y aplanamos de vuelta a (N * K)
    adv_matrix = returns_matrix - baseline
    advantages = adv_matrix.view(total_envs)

    # 2. Normalización por combo (S, H)
    # Necesitamos agrupar los índices de los entornos por su combo
    combo_to_indices = {}
    for env_idx, combo in enumerate(combo_ids):
        if combo not in combo_to_indices:
            combo_to_indices[combo] = []
        combo_to_indices[combo].append(env_idx)

    for combo, indices in combo_to_indices.items():
        indices_tensor = torch.tensor(indices, dtype=torch.long, device=device)
        combo_adv = advantages[indices_tensor]

        std = combo_adv.std()
        # Solo dividimos si hay desviación (evitamos dividir por cero si todos empatan)
        if std > 1e-4:
            advantages[indices_tensor] = combo_adv / std
        else:
            # Si todos los rollouts de este combo sacaron lo mismo, la ventaja es 0
            advantages[indices_tensor] = 0.0

    # 3. Clipping para proteger contra outliers
    advantages = torch.clamp(advantages, min=-clip_val, max=clip_val)

    # 4. Propagación temporal
    # Expandimos las ventajas de los entornos (N * K) a cada paso individual dado
    advantages_history = advantages[rollout_idx_history]

    return advantages_history

def compute_pomo_loss(
    logits: torch.Tensor,
    ref_logits: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    ent_coef: float,
    kl_coef: float
):
    """
    Calcula la función objetivo completa de la Fase 2:
    L = - (Adv * log π(a|s)) - ent_coef * H(π) + kl_coef * KL(π || π_ref)

    Args:
        logits: Tensores de salida del modelo actual en el paso de replay. (Batch, Num_Acciones)
        ref_logits: Tensores de salida del modelo congelado. (Batch, Num_Acciones)
        actions: Las acciones reales que tomó el muestreo. (Batch,)
        advantages: Las ventajas POMO asociadas a cada estado. (Batch,)
    """
    # 1. Asegurar precisión float32 para la estabilidad del log_softmax
    logits = logits.float()
    ref_logits = ref_logits.float()

    # Calculamos log_probs y probs del modelo actual
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)

    # Calculamos log_probs del modelo de referencia
    with torch.no_grad():
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

    # ==========================================================
    # TÉRMINO 1: Policy Gradient
    # ==========================================================
    # Extraemos el log(prob) solo de la acción que se tomó
    # Usamos gather para indexar de forma vectorizada
    log_prob_actions = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

    # pg_loss = - (Ventaja * log_prob)
    pg_loss = -(advantages * log_prob_actions).mean()

    # ==========================================================
    # TÉRMINO 2: Bono de Entropía
    # ==========================================================
    # H(π) = - Σ (p * log p)
    # Gracias a la máscara de -1e4, p será 0.0 y log(p) será ~-10000,
    # dando 0.0 exacto en lugar de NaN.
    entropy = -(probs * log_probs).sum(dim=-1).mean()

    # ==========================================================
    # TÉRMINO 3: Ancla KL (Kullback-Leibler Divergence)
    # ==========================================================
    # KL(π || π_ref) = Σ (p * (log p - log p_ref))
    kl_div = (probs * (log_probs - ref_log_probs)).sum(dim=-1).mean()

    # ==========================================================
    # PÉRDIDA TOTAL
    # ==========================================================
    total_loss = pg_loss - (ent_coef * entropy) + (kl_coef * kl_div)

    return total_loss, pg_loss.detach(), entropy.detach(), kl_div.detach()

def update_model_pomo(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    states_history: list,
    actions_history: torch.Tensor,
    advantages_history: torch.Tensor,
    minibatch_size: int,
    ent_coef: float,
    kl_coef: float,
    grad_clip: float,
    device: torch.device
):
    """
    Realiza el forward plano acumulando gradientes en chunks de minibatch_size.
    Aplica clip_grad_norm_ y da un solo paso de optimización.
    """
    model.train()
    ref_model.eval()

    total_steps = len(actions_history)
    num_inputs = len(states_history[0])
    stacked_states = []

    for j in range(num_inputs):
        j_tensors = [states_history[i][j] for i in range(total_steps)]
        stacked_states.append(torch.stack(j_tensors))

    total_pg_loss = 0.0
    total_entropy = 0.0
    total_kl = 0.0

    optimizer.zero_grad(set_to_none=True)
    indices = torch.randperm(total_steps, device=device)

    for start_idx in range(0, total_steps, minibatch_size):
        end_idx = min(start_idx + minibatch_size, total_steps)
        mb_indices = indices[start_idx:end_idx]
        mb_size = end_idx - start_idx

        mb_states = [stacked_states[j][mb_indices] for j in range(num_inputs)]
        mb_actions = actions_history[mb_indices]
        mb_advantages = advantages_history[mb_indices]

        # Inferencia directa en float32
        logits = model(*mb_states)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]

        with torch.no_grad():
            ref_logits = ref_model(*mb_states)
            if isinstance(ref_logits, (list, tuple)):
                ref_logits = ref_logits[0]

        loss, pg, ent, kl = compute_pomo_loss(
            logits, ref_logits, mb_actions, mb_advantages, ent_coef, kl_coef
        )

        ratio = mb_size / total_steps
        scaled_loss = loss * ratio

        # Backward directo sin escalador
        scaled_loss.backward()

        total_pg_loss += pg.item() * ratio
        total_entropy += ent.item() * ratio
        total_kl += kl.item() * ratio

    # Clipping y step clásicos
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    return total_pg_loss, total_entropy, total_kl

@torch.no_grad()
def evaluate_dev_greedy(
    model: torch.nn.Module,
    dev_instances: List[Tuple[Any, int]],
    max_steps_dict: dict,
    input_adapter_config: tuple,
    device: torch.device
) -> Tuple[float, int, int]:
    """
    Evalúa el modelo de forma puramente determinista (greedy).
    Retorna: (promedio de pasos penalizados, instancias resueltas, total de instancias)
    """
    model.eval()
    la_class, *la_args = input_adapter_config

    total_envs = len(dev_instances)
    layouts = []
    max_steps_array = []
    
    # NUEVO: Lista de sets para guardar el historial completo de estados de cada layout
    visited_states_list = [set() for _ in range(total_envs)]

    for inst in dev_instances:
        layouts.append(copy.deepcopy(inst))
        combo = (len(inst.stacks), inst.H)
        max_steps_array.append(max_steps_dict.get(combo, 150))

    active_mask = np.ones(total_envs, dtype=bool)
    steps_taken = np.zeros(total_envs, dtype=int)

    while np.any(active_mask):
        active_indices = np.where(active_mask)[0]
        
        # NUEVO: Guardar el estado actual antes de mover
        for idx in active_indices:
            current_state = tuple(tuple(stack) for stack in layouts[idx].stacks)
            visited_states_list[idx].add(current_state)

        input_adapter = la_class(*la_args)
        for idx in active_indices:
            lay = layouts[idx]
            input_adapter.add(input_adapter.input_2_vec(lay, lay.H))

        batch_dict = input_adapter.get()
        batch_inputs = [torch.tensor(batch_dict[k], device=device) for k in batch_dict]

        logits = model(*batch_inputs)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]

        mask = torch.full_like(logits, -1e4)
        for i, idx in enumerate(active_indices):
            for src, dst in get_feasible_moves(layouts[idx]):
                action_idx = src * (input_adapter.S_max - 1) + (dst - int(dst > src))
                mask[i, action_idx] = 0.0

        masked_logits = logits + mask
        
        # NUEVO: Ordenamos todas las acciones posibles de mejor a peor
        top_values_batch, top_indices_batch = torch.sort(masked_logits, dim=-1, descending=True)

        for i, idx in enumerate(active_indices):
            lay = layouts[idx]
            top_indices = top_indices_batch[i]
            top_values = top_values_batch[i]
            visited_states = visited_states_list[idx]
            
            action_applied = False
            
            # NUEVO: Buscamos la mejor acción que no nos lleve a un estado repetido
            for j in range(len(top_indices)):
                best_index = top_indices[j].item()
                best_value = top_values[j].item()
                
                # Si llegamos a los enmascarados, no hay más movimientos válidos
                if best_value <= -1e4:
                    break
                    
                src = best_index // (input_adapter.S_max - 1)
                rem = best_index % (input_adapter.S_max - 1)
                dst = rem + 1 if rem >= src else rem
                
                # Previsualización
                temp_layout = copy.deepcopy(lay)
                temp_layout.move(src, dst)
                next_state = tuple(tuple(stack) for stack in temp_layout.stacks)
                
                if next_state not in visited_states:
                    lay.move(src, dst)
                    steps_taken[idx] += 1
                    action_applied = True
                    break
            
            # Si ninguna acción fue válida (todas ciclan o no hay factibles), abortamos este layout
            if not action_applied:
                active_mask[idx] = False
            else:
                if lay.is_sorted() or steps_taken[idx] >= max_steps_array[idx]:
                    active_mask[idx] = False

    solved_count = sum(1 for lay in layouts if lay.is_sorted())

    penalized_steps = [
        steps_taken[i] if layouts[i].is_sorted() else max_steps_array[i]
        for i in range(total_envs)
    ]

    return float(np.mean(penalized_steps)), solved_count, total_envs

def train_pomo_rl(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    generators: list,
    train_instances_dict: dict,
    dev_instances: list,
    pomo_config,
    max_steps_dict: dict,
    input_adapter_config: tuple,
    device: torch.device,
    save_dir: str,
    model_name: str
):
    print(f"Iniciando POMO RL por {pomo_config.updates} updates...")

    os.makedirs(save_dir, exist_ok=True)
    best_dev_score = float('inf')
    history = []

    for update in range(1, pomo_config.updates + 1):
        start_time = time.time()

        progress = (update - 1) / max(1, (pomo_config.updates - 1))
        ent_coef = pomo_config.ent_coef_start - progress * (pomo_config.ent_coef_start - pomo_config.ent_coef_end)

        # 2. Muestreo Aleatorio del Pool Pre-Generado
        instances = []
        for generator in generators:
            # Tomamos 'instances_per_combo' de las pre-generadas
            combo_instances = random.sample(train_instances_dict[generator], pomo_config.instances_per_combo)
            instances.extend(combo_instances)

        states_hist, actions_hist, returns, combo_ids, rollout_idx_hist, is_solved = sample_pomo_rollouts(
            model, instances,
            pomo_config.k_rollouts,
            max_steps_dict,
            input_adapter_config,
            pomo_config.unsolved_extra,
            device
        )

        # 3. Cálculo de Ventajas
        advantages_hist = compute_pomo_advantages(
            returns, combo_ids, rollout_idx_hist,
            pomo_config.k_rollouts,
            pomo_config.adv_clip,
            device
        )

        # 4. Update On-Policy (Con Gradiente)
        pg_loss, entropy, kl_div = update_model_pomo(
            model, ref_model, optimizer,
            states_hist, actions_hist, advantages_hist,
            pomo_config.minibatch_size, ent_coef, pomo_config.kl_coef,
            pomo_config.grad_clip, device
        )

        # 5. Telemetría
        solved_rate = is_solved.float().mean().item() * 100
        avg_steps = -returns[is_solved].mean().item() if solved_rate > 0 else 0.0

        fps = len(actions_hist) / max(1e-4, time.time() - start_time)

        # Logging
        print(f"Update {update:05d} | "
              f"PG: {pg_loss:+.3f} | Ent: {entropy:.3f} | KL: {kl_div:.3f} | "
              f"Solve: {solved_rate:3.0f}% | AvgSteps: {avg_steps:.1f} | "
              f"EntCoef: {ent_coef:.4f} | {fps:.0f} steps/s")

        history.append({
            "update": update, "pg_loss": pg_loss, "entropy": entropy,
            "kl_div": kl_div, "solve_rate": solved_rate
        })

#        6. Evaluación en Dev-Set
        if update % pomo_config.eval_interval == 0:
            dev_score, solved_count, total_dev = evaluate_dev_greedy(
                model, dev_instances, max_steps_dict,
                input_adapter_config, device
            )
            solve_rate_dev = (solved_count / total_dev) * 100

            print(f"   >>> 🧪 Evaluación Dev: {dev_score:.2f} pasos | Resueltas: {solved_count}/{tot al_dev} ({solve_rate_dev:.1f}%) | (Mejor: {min(dev_score, best_dev_score):.2f})")

            if dev_score < best_dev_score:
                best_dev_score = dev_score
                save_model(model, model_name)
                print(f"   >>> 🏆 ¡Nuevo mejor modelo guardado!")

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

    # Generación offline antes de empezar
    print("Generando pool estático de instancias de entrenamiento y validación...")
    for gen in generators:
        # Tuplas (instancia, H) para validación
        combo_devs = gen.generate_instances(test_size_per_combo)
        dev_instances.extend([Layout(inst, gen.H) for inst in combo_devs])

        # Tuplas (instancia, H) para entrenamiento
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
