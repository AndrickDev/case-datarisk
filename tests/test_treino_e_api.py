import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import criar_app
from src.config import FEATURES
from src.treino import NOME_MODELO, carregar, prever, salvar, treinar


@pytest.fixture(scope="module")
def pipeline(desenvolvimento):
    return treinar(desenvolvimento)


@pytest.fixture(scope="module")
def caminho_modelo(pipeline, tmp_path_factory):
    return salvar(pipeline, tmp_path_factory.mktemp("modelo"), n_linhas=1)


def test_serializacao_joblib_preserva_previsoes(pipeline, caminho_modelo, desenvolvimento):
    recarregado = carregar(caminho_modelo)
    amostra = desenvolvimento.head(50)
    assert caminho_modelo.name == NOME_MODELO
    np.testing.assert_allclose(prever(pipeline, amostra), prever(recarregado, amostra))


def test_pipeline_expoe_a_lista_unica_de_features(pipeline):
    assert sorted(pipeline.named_steps["prep"].get_feature_names_out()) == sorted(FEATURES)


def test_api_saude(caminho_modelo):
    cliente = TestClient(criar_app(caminho_modelo))
    r = cliente.get("/saude")
    assert r.status_code == 200
    assert r.json()["features"] == FEATURES


def test_api_score_retorna_probabilidades_validas(caminho_modelo):
    cliente = TestClient(criar_app(caminho_modelo))
    corpo = {"registros": [
        {"DATA_EMISSAO_DOCUMENTO": "2019-05-10", "DATA_VENCIMENTO": "2019-06-10",
         "VALOR_A_PAGAR": 12000.0, "TAXA": 6.99, "PORTE": "MEDIO", "DDD": 31},
        {"DATA_EMISSAO_DOCUMENTO": "2019-05-12", "DATA_VENCIMENTO": "2019-06-20"},
    ]}
    r = cliente.post("/score", json=corpo)
    assert r.status_code == 200
    prob = r.json()["probabilidades"]
    assert len(prob) == 2 and all(0.0 <= p <= 1.0 for p in prob)


def test_api_rejeita_corpo_invalido(caminho_modelo):
    cliente = TestClient(criar_app(caminho_modelo))
    assert cliente.post("/score", json={"registros": []}).status_code == 422
    assert cliente.post("/score", json={"registros": [{"TAXA": 1}]}).status_code == 422


def test_api_sem_modelo_retorna_503(tmp_path):
    cliente = TestClient(criar_app(tmp_path / "inexistente.joblib"))
    corpo = {"registros": [{"DATA_EMISSAO_DOCUMENTO": "2019-05-10", "DATA_VENCIMENTO": "2019-06-10"}]}
    assert cliente.post("/score", json=corpo).status_code == 503
