"""Leitura das bases e construcao do conjunto de modelagem."""
from pathlib import Path

import pandas as pd

from src.config import COL_ALVO, COL_ID, COL_SAFRA, DATA_DIR, LIMITE_ATRASO_DIAS

ARQUIVOS = {
    "cadastral": "base_cadastral.csv",
    "info": "base_info.csv",
    "pagamentos": "base_pagamentos_desenvolvimento.csv",
    "teste": "base_pagamentos_teste.csv",
}


def carregar_bases(data_dir=None):
    pasta = Path(data_dir) if data_dir else DATA_DIR
    return {nome: pd.read_csv(pasta / arq, sep=";") for nome, arq in ARQUIVOS.items()}


def derivar_variaveis(df):
    """Variaveis derivadas apenas de informacao disponivel na emissao do documento."""
    df = df.copy()
    emissao = pd.to_datetime(df["DATA_EMISSAO_DOCUMENTO"], errors="coerce")
    vencimento = pd.to_datetime(df["DATA_VENCIMENTO"], errors="coerce")
    df["MES_EMISSAO"] = emissao.dt.month
    df["DIA_EMISSAO"] = emissao.dt.day
    df["DIA_VENCIMENTO"] = vencimento.dt.day
    df["PRAZO_DIAS"] = (vencimento - emissao).dt.days
    if "DATA_CADASTRO" in df.columns:
        cadastro = pd.to_datetime(df["DATA_CADASTRO"], errors="coerce")
        df["TEMPO_CADASTRO_DIAS"] = (emissao - cadastro).dt.days
    return df


def _juntar(pagamentos, cadastral, info):
    base = pagamentos.merge(cadastral, on=COL_ID, how="left")
    base = base.merge(info, on=[COL_ID, COL_SAFRA], how="left")
    return derivar_variaveis(base)


def construir_alvo(pagamentos, limite=LIMITE_ATRASO_DIAS):
    """INADIMPLENTE = 1 quando o pagamento ocorreu com atraso >= limite dias."""
    out = pagamentos.copy()
    pago = pd.to_datetime(out["DATA_PAGAMENTO"], errors="coerce")
    venc = pd.to_datetime(out["DATA_VENCIMENTO"], errors="coerce")
    out["DIAS_ATRASO"] = (pago - venc).dt.days
    out[COL_ALVO] = (out["DIAS_ATRASO"] >= limite).astype(int)
    return out


def montar_desenvolvimento(bases):
    pag = construir_alvo(bases["pagamentos"])
    return _juntar(pag, bases["cadastral"], bases["info"]).reset_index(drop=True)


def montar_teste(bases):
    return _juntar(bases["teste"], bases["cadastral"], bases["info"]).reset_index(drop=True)
