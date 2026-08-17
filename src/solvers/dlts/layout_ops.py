"""
Operaciones sobre Layout que el propio cpmp/layout.py no ofrece de forma segura:
Layout.prev()/next() están rotos (usan variables no definidas y no restauran
sorted_elements/sorted_stack/unsorted_stacks/full_stacks), así que DFS/LDS
necesitan su propio undo. No se modifica cpmp/layout.py.

undo_last_move recomputa los atributos derivados desde stacks en vez de invertir
los deltas incrementales que mantiene Layout.move: es O(S*H) (~decenas de ops,
despreciable frente a un forward de la red) y, a diferencia de invertir a mano
los deltas, es correcto por construcción (replica exactamente lo que haría
Layout.__init__ sobre el estado ya revertido).
"""
from cpmp.layout import Layout, compute_sorted_elements


def recompute_derived(layout: Layout) -> None:
    """Reconstruye sorted_elements, sorted_stack, unsorted_stacks, full_stacks
    y total_elements desde layout.stacks, replicando exactamente la lógica de
    Layout.__init__ (incluyendo que el guard `j < len(sorted_stack)` de
    is_sorted_stack nunca dispara durante una construcción/reconstrucción
    limpia, por eso no hace falta llamar a is_sorted_stack aquí)."""
    stacks = layout.stacks
    H = layout.H

    sorted_elements = []
    sorted_stack = []
    unsorted_stacks = 0
    full_stacks = 0
    total_elements = 0

    for stack in stacks:
        total_elements += len(stack)
        if len(stack) == H:
            full_stacks += 1
        se = compute_sorted_elements(stack)
        sorted_elements.append(se)
        is_sorted = len(stack) == se
        if is_sorted:
            sorted_stack.append(True)
        else:
            unsorted_stacks += 1
            sorted_stack.append(False)

    layout.sorted_elements = sorted_elements
    layout.total_elements = total_elements
    layout.sorted_stack = sorted_stack
    layout.unsorted_stacks = unsorted_stacks
    layout.full_stacks = full_stacks


def undo_last_move(layout: Layout) -> tuple[int, int]:
    """Deshace layout.moves[-1] (siempre un movimiento tope-a-tope, index=-1,
    que es lo único que produce nuestra búsqueda). Devuelve (src, dst)."""
    if not layout.moves:
        raise IndexError("undo_last_move: no hay movimientos que deshacer")

    src, dst = layout.moves[-1]
    c = layout.stacks[dst].pop()
    layout.stacks[src].append(c)
    layout.moves.pop()
    layout.steps -= 1
    layout.current_step -= 1
    recompute_derived(layout)
    return src, dst


def apply_move(layout: Layout, src: int, dst: int) -> None:
    layout.move(src, dst)


def state_key(layout: Layout) -> tuple:
    return tuple(tuple(stack) for stack in layout.stacks)


def clone_layout(layout: Layout) -> Layout:
    """Copia manual (~3x más rápida que copy.deepcopy porque no pasa por el
    protocolo genérico de pickle/introspección) usada en el hot path de LDS,
    donde cada heappop clona el layout del padre."""
    new = Layout.__new__(Layout)
    new.stacks = [list(s) for s in layout.stacks]
    new.H = layout.H
    new.sorted_elements = list(layout.sorted_elements)
    new.total_elements = layout.total_elements
    new.sorted_stack = list(layout.sorted_stack)
    new.unsorted_stacks = layout.unsorted_stacks
    new.steps = layout.steps
    new.current_step = layout.current_step
    new.moves = list(layout.moves)
    new.full_stacks = layout.full_stacks
    new.last_sd = layout.last_sd
    new.bsg_moves = list(layout.bsg_moves)
    return new
