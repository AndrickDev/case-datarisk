"""Auditoria de vazamento do LightGBM final do notebook.

Reproduz a montagem de X do notebook (colunas numericas de `pag`, sem as
colunas de desfecho), lista X.columns e compara o AUC com e sem ID_CLIENTE.
Uso: python -m src.auditoria
"""
import json

import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.config import COL_ALVO, COL_ID, PARAMS_LGBM, REPORTS_DIR, SEMENTE
from src.data import carregar_bases, construir_alvo


def montar_x_notebook(pagamentos):
    """Replica a celula de treino do notebook, que parte de `pag` e filtra por dtype."""
    pag = construir_alvo(pagamentos)
    emissao = pd.to_datetime(pag["DATA_EMISSAO_DOCUMENTO"])
    pag["MES_EMISSAO"] = emissao.dt.month
    pag["DIA_EMISSAO"] = emissao.dt.day
    pag["DIA_VENCIMENTO"] = pd.to_datetime(pag["DATA_VENCIMENTO"]).dt.day
    remover = [COL_ALVO, "DIAS_ATRASO", "DATA_PAGAMENTO", "DATA_VENCIMENTO", "DATA_EMISSAO_DOCUMENTO"]
    X = pag.drop(columns=remover).select_dtypes(include=["int", "float", "bool"])
    return X, pag[COL_ALVO]


def auc_cv(X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMENTE)
    s = cross_val_score(lgb.LGBMClassifier(**PARAMS_LGBM), X, y, cv=cv, scoring="roc_auc")
    return float(s.mean()), float(s.std())


def executar():
    bases = carregar_bases()
    X, y = montar_x_notebook(bases["pagamentos"])
    colunas = list(X.columns)
    print("X.columns:", colunas)
    media_com, dp_com = auc_cv(X, y)
    media_sem, dp_sem = auc_cv(X.drop(columns=[COL_ID]), y)
    resultado = {
        "colunas_x": colunas,
        "identificador_presente": COL_ID in colunas,
        "auc_com_id_cliente": {"media": media_com, "desvio": dp_com},
        "auc_sem_id_cliente": {"media": media_sem, "desvio": dp_sem},
        "validacao": "StratifiedKFold(5, shuffle=True, random_state=42), sem agrupamento",
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "auditoria_vazamento.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"AUC com ID_CLIENTE: {media_com:.4f} +- {dp_com:.4f}")
    print(f"AUC sem ID_CLIENTE: {media_sem:.4f} +- {dp_sem:.4f}")
    return resultado


if __name__ == "__main__":
    executar()
