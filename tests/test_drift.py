import numpy as np

from src.drift import classificar, psi_categorico, psi_numerico


def test_psi_numerico_de_amostras_identicas_e_zero():
    x = np.random.default_rng(0).normal(size=5000)
    assert psi_numerico(x, x) < 1e-9


def test_psi_numerico_detecta_deslocamento_forte():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 5000)
    atual = rng.normal(2, 1, 5000)
    assert psi_numerico(ref, atual) > 0.25


def test_psi_categorico_trata_categoria_nova():
    ref = ["A"] * 80 + ["B"] * 20
    atual = ["A"] * 40 + ["B"] * 20 + ["C"] * 40
    assert psi_categorico(ref, atual) > 0.25


def test_classificacao_por_limites():
    assert classificar(0.05) == "estavel"
    assert classificar(0.15) == "atencao"
    assert classificar(0.30) == "alerta"
