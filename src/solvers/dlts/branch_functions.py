"""
Funciones de poda de anchura (MP, §4.4 del paper DLTS). Port 1:1 de
DLTS-master/tree_search.py (branch_strategy_*) del original, mismos nombres
de parámetros para poder difear/testear contra el original.

mp -- probabilidad máxima que sale de la red de política en el nodo
ub -- coste de la mejor solución conocida
dd -- profundidad actual (Cost(n))
pp -- parámetro de anchura p

Casos borde deliberadamente NO corregidos (son parte del método, no bugs):
- mp_log con dd=0 (raíz) da -inf -> no poda nada en la raíz.
- mp_linear con dd=0 da mp -> solo sobrevive el argmax en la raíz.
- mp_quadratic con dd=0 coincide con mp_constant.
"""
from typing import Callable
import numpy as np


def mp_constant(mp, ub, dd, pp):
    return mp * (1 - pp)


def mp_linear(mp, ub, dd, pp):
    return mp * (1 - pp * (dd / ub))


def mp_quadratic(mp, ub, dd, pp):
    return mp * (1 - pp * ((ub - dd) ** 2 / ub ** 2))


def mp_log(mp, ub, dd, pp):
    with np.errstate(divide='ignore'):
        return mp * (1 - pp * (-np.log(dd / ub)))


BRANCH_FUNCS: dict[str, Callable] = {
    "constant": mp_constant,
    "linear": mp_linear,
    "quadratic": mp_quadratic,
    "log": mp_log,
}


def get_branch_func(name: str) -> Callable:
    if name not in BRANCH_FUNCS:
        raise ValueError(f"bstrategy desconocida: {name!r}. Opciones: {list(BRANCH_FUNCS)}")
    return BRANCH_FUNCS[name]
