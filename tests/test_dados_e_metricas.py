import numpy as np
import pandas as pd

from src import config
from src.auditoria import montar_x_notebook
from src.data import construir_alvo
from src.metrics import gini, ks, resumo


def test_alvo_usa_limite_de_cinco_dias():
    pag = pd.DataFrame({
        "DATA_VENCIMENTO": ["2020-01-10"] * 3,
        "DATA_PAGAMENTO": ["2020-01-14", "2020-01-15", "2020-01-08"],
    })
    assert construir_alvo(pag)["INADIMPLENTE"].tolist() == [0, 1, 0]


def test_features_nao_contem_identificadores():
    assert not set(config.FEATURES) & set(config.COLUNAS_PROIBIDAS)


def test_auditoria_detecta_id_cliente_em_x(pagamentos_sinteticos):
    X, _ = montar_x_notebook(pagamentos_sinteticos["pagamentos"])
    assert config.COL_ID in X.columns


def test_metricas_separacao_perfeita():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert resumo(y, s) == {"auc": 1.0, "ks": 1.0, "gini": 1.0}


def test_escore_aleatorio_tem_gini_e_ks_proximos_de_zero():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 20000)
    s = rng.random(20000)
    assert abs(gini(y, s)) < 0.05
    assert ks(y, s) < 0.05
