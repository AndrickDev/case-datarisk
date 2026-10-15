import numpy as np
import pandas as pd

from src.explicabilidade import TransformadorWOE, classificar_iv, tabela_woe


def test_iv_de_variavel_independente_do_alvo_e_baixo():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"VALOR_A_PAGAR": rng.normal(size=20000)})
    y = pd.Series(rng.integers(0, 2, 20000))
    t = TransformadorWOE(["VALOR_A_PAGAR"]).fit(X, y)
    assert t.iv_["VALOR_A_PAGAR"] < 0.02


def test_iv_de_variavel_preditiva_e_alto_e_woe_cresce_com_risco():
    rng = np.random.default_rng(1)
    x = rng.normal(size=20000)
    y = pd.Series((rng.random(20000) < 1 / (1 + np.exp(-(2 * x - 3)))).astype(int))
    X = pd.DataFrame({"VALOR_A_PAGAR": x})
    t = TransformadorWOE(["VALOR_A_PAGAR"]).fit(X, y)
    woe = t.transform(X)["VALOR_A_PAGAR"]
    assert t.iv_["VALOR_A_PAGAR"] > 0.30
    assert np.corrcoef(x, woe)[0, 1] > 0.8


def test_tabela_woe_suaviza_faixa_sem_eventos():
    faixa = pd.Series(["a", "a", "b", "b"])
    y = pd.Series([0, 0, 1, 0])
    woe, iv = tabela_woe(faixa, y)
    assert np.isfinite(list(woe.values())).all()
    assert iv > 0


def test_classes_de_iv():
    assert classificar_iv(0.01) == "nao preditiva"
    assert classificar_iv(0.5) == "forte"
