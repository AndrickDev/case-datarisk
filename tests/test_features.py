import numpy as np
import pandas as pd
import pytest

from src import config
from src.features import construir_pipeline, preparar_entrada


def test_preparar_entrada_usa_exatamente_a_lista_unica(desenvolvimento):
    X = preparar_entrada(desenvolvimento)
    assert list(X.columns) == config.FEATURES


def test_lista_com_identificador_e_rejeitada(desenvolvimento):
    with pytest.raises(ValueError):
        preparar_entrada(desenvolvimento, features=config.FEATURES + [config.COL_ID])


def test_treino_e_inferencia_usam_as_mesmas_colunas(desenvolvimento):
    X = preparar_entrada(desenvolvimento)
    y = desenvolvimento[config.COL_ALVO]
    pipe = construir_pipeline().fit(X, y)
    colunas_treino = list(pipe.named_steps["prep"].get_feature_names_out())
    novo = desenvolvimento.drop(columns=["PORTE", "DDD"]).head(5)
    prob = pipe.predict_proba(preparar_entrada(novo))[:, 1]
    assert sorted(colunas_treino) == sorted(config.FEATURES)
    assert prob.shape == (5,)


def test_categoria_inedita_e_nulos_nao_quebram_inferencia(desenvolvimento):
    X = preparar_entrada(desenvolvimento)
    pipe = construir_pipeline().fit(X, desenvolvimento[config.COL_ALVO])
    novo = X.head(3).copy()
    novo["PORTE"] = "NOVO"
    novo["RENDA_MES_ANTERIOR"] = np.nan
    prob = pipe.predict_proba(novo)[:, 1]
    assert np.all((prob >= 0) & (prob <= 1))
    assert not np.isnan(prob).any()


def test_codificacao_categorica_nao_depende_do_lote():
    a = preparar_entrada(pd.DataFrame({"DDD": [31.0, np.nan]}), features=["DDD"])
    b = preparar_entrada(pd.DataFrame({"DDD": [31]}), features=["DDD"])
    assert a["DDD"].iloc[0] == b["DDD"].iloc[0] == "31"
    assert a["DDD"].iloc[1] == "NA"
