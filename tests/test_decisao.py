import numpy as np

from src.decisao import (
    ajustar_isotonica,
    ajustar_platt,
    limiar_por_custo,
    matriz_confusao,
    precisao_recall,
)


def test_matriz_confusao_e_precisao_recall():
    y = np.array([1, 1, 0, 0, 1, 0])
    p = np.array([0.9, 0.4, 0.8, 0.1, 0.7, 0.2])
    m = matriz_confusao(y, p, 0.5)
    assert m == {"tp": 2, "fp": 1, "fn": 1, "tn": 2}
    assert precisao_recall(m) == (2 / 3, 2 / 3)


def test_limiar_por_custo_diminui_quando_falso_negativo_e_mais_caro():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 5000)
    p = np.clip(0.5 * y + rng.normal(0.25, 0.2, 5000), 0, 1)
    t_igual = limiar_por_custo(y, p, custo_fn=1, custo_fp=1)
    t_fn_caro = limiar_por_custo(y, p, custo_fn=10, custo_fp=1)
    assert t_fn_caro < t_igual


def test_calibradores_retornam_probabilidades_validas():
    rng = np.random.default_rng(4)
    score = rng.random(3000)
    y = (rng.random(3000) < score**2).astype(int)
    for ajustar in (ajustar_platt, ajustar_isotonica):
        f = ajustar(score, y)
        out = f(np.array([0.0, 0.5, 1.0]))
        assert np.all((out >= 0) & (out <= 1))
