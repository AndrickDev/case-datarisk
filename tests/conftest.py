import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def pagamentos_sinteticos():
    """Bases sinteticas com a mesma estrutura das bases do desafio."""
    rng = np.random.default_rng(0)
    n = 1500
    clientes = rng.integers(1, 80, n)
    inicio = pd.Timestamp("2019-01-01")
    emissao = inicio + pd.to_timedelta(rng.integers(0, 330, n), unit="D")
    venc = emissao + pd.to_timedelta(rng.integers(10, 60, n), unit="D")
    valor = rng.lognormal(9, 1, n)
    atraso = rng.integers(-3, 4, n) + (valor > np.quantile(valor, 0.8)) * rng.integers(0, 15, n)
    df = pd.DataFrame({
        "ID_CLIENTE": clientes,
        "SAFRA_REF": emissao.strftime("%Y-%m"),
        "DATA_EMISSAO_DOCUMENTO": emissao.strftime("%Y-%m-%d"),
        "DATA_VENCIMENTO": venc.strftime("%Y-%m-%d"),
        "DATA_PAGAMENTO": (venc + pd.to_timedelta(atraso, unit="D")).strftime("%Y-%m-%d"),
        "VALOR_A_PAGAR": valor,
        "TAXA": rng.choice([5.0, 6.99, 8.5], n),
    })
    cad = pd.DataFrame({
        "ID_CLIENTE": np.arange(1, 80),
        "DATA_CADASTRO": "2015-06-01",
        "DDD": rng.choice([11, 31, 75, np.nan], 79),
        "FLAG_PF": np.nan,
        "SEGMENTO_INDUSTRIAL": rng.choice(["Comercio", "Servicos", np.nan], 79),
        "DOMINIO_EMAIL": rng.choice(["GMAIL", "HOTMAIL", np.nan], 79),
        "PORTE": rng.choice(["PEQUENO", "MEDIO", "GRANDE", np.nan], 79),
        "CEP_2_DIG": rng.integers(10, 99, 79),
    })
    info = df[["ID_CLIENTE", "SAFRA_REF"]].drop_duplicates().reset_index(drop=True)
    info["RENDA_MES_ANTERIOR"] = rng.lognormal(11, 1, len(info))
    info["NO_FUNCIONARIOS"] = rng.integers(1, 300, len(info)).astype(float)
    return {"pagamentos": df, "cadastral": cad, "info": info}


@pytest.fixture(scope="session")
def desenvolvimento(pagamentos_sinteticos):
    from src.data import montar_desenvolvimento
    return montar_desenvolvimento(pagamentos_sinteticos)
