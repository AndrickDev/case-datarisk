"""Auditoria de vazamento do LightGBM final do notebook.

Reproduz a montagem de X do notebook (colunas numericas de `pag`, sem as
colunas de desfecho), lista X.columns e compara o AUC com e sem ID_CLIENTE.
Uso: python -m src.auditoria
"""
import json

import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score

from src.config import COL_ALVO, COL_ID, COL_SAFRA, PARAMS_LGBM, REPORTS_DIR, SEMENTE
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


def auc_grupo(X, y, grupos):
    s = cross_val_score(lgb.LGBMClassifier(**PARAMS_LGBM), X, y, cv=GroupKFold(5), groups=grupos, scoring="roc_auc")
    return float(s.mean()), float(s.std())


def auc_out_of_time(X, y, safras, k=6):
    """Treina nas safras anteriores e avalia nas k mais recentes (mesmo corte de src.validacao)."""
    from src.metrics import auc
    from src.validacao import split_out_of_time

    tr, te = split_out_of_time(safras, k)
    modelo = lgb.LGBMClassifier(**PARAMS_LGBM).fit(X.iloc[tr], y.iloc[tr])
    return float(auc(y.iloc[te], modelo.predict_proba(X.iloc[te])[:, 1]))


def executar():
    bases = carregar_bases()
    X, y = montar_x_notebook(bases["pagamentos"])
    colunas = list(X.columns)
    print("X.columns:", colunas)
    media_com, dp_com = auc_cv(X, y)
    media_sem, dp_sem = auc_cv(X.drop(columns=[COL_ID]), y)
    pag = bases["pagamentos"]
    grupos = pag[COL_ID].to_numpy()
    safras = pag[COL_SAFRA].tolist()
    sem_id = X.drop(columns=[COL_ID])
    g_com, _ = auc_grupo(X, y, grupos)
    g_sem, _ = auc_grupo(sem_id, y, grupos)
    resultado = {
        "colunas_x": colunas,
        "identificador_presente": COL_ID in colunas,
        "auc_com_id_cliente": {"media": media_com, "desvio": dp_com},
        "auc_sem_id_cliente": {"media": media_sem, "desvio": dp_sem},
        "validacao": "StratifiedKFold(5, shuffle=True, random_state=42), sem agrupamento",
        "auc_grupo_cliente": {"com_id_cliente": g_com, "sem_id_cliente": g_sem},
        "auc_out_of_time_k6": {
            "com_id_cliente": auc_out_of_time(X, y, safras),
            "sem_id_cliente": auc_out_of_time(sem_id, y, safras),
        },
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "auditoria_vazamento.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"AUC com ID_CLIENTE: {media_com:.4f} +- {dp_com:.4f}")
    print(f"AUC sem ID_CLIENTE: {media_sem:.4f} +- {dp_sem:.4f}")
    print("AUC por grupo de cliente:", resultado["auc_grupo_cliente"])
    print("AUC out-of-time k=6:", resultado["auc_out_of_time_k6"])
    return resultado


if __name__ == "__main__":
    executar()
