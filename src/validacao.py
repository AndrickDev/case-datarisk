"""Esquemas de validacao: aleatoria, por grupo de cliente e out-of-time por safra.

Uso: python -m src.validacao
"""
import json

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, StratifiedKFold

from src.config import COL_ALVO, COL_ID, COL_SAFRA, FEATURES, REPORTS_DIR, SEMENTE
from src.data import carregar_bases, montar_desenvolvimento
from src.features import construir_pipeline, preparar_entrada
from src.metrics import resumo

# Conjunto reproduzido do notebook (sem o identificador), usado como referencia.
FEATURES_NOTEBOOK = ["VALOR_A_PAGAR", "TAXA", "MES_EMISSAO", "DIA_EMISSAO", "DIA_VENCIMENTO"]
FEATURES_SEM_TAXA_VALOR = [f for f in FEATURES if f not in ("TAXA", "VALOR_A_PAGAR")]

CONJUNTOS = {
    "notebook_sem_id": FEATURES_NOTEBOOK,
    "completo": FEATURES,
    "completo_sem_taxa_valor": FEATURES_SEM_TAXA_VALOR,
}


def split_out_of_time(safras, k):
    """Treino nas safras anteriores e teste nas k safras mais recentes."""
    ordenadas = sorted(set(safras))
    if k >= len(ordenadas):
        raise ValueError("k deve ser menor que o numero de safras")
    teste = set(ordenadas[-k:])
    mascara_teste = np.array([s in teste for s in safras])
    return np.where(~mascara_teste)[0], np.where(mascara_teste)[0]


def _medio(lista):
    chaves = lista[0].keys()
    return {c: float(np.mean([m[c] for m in lista])) for c in chaves}


def _desvio(lista):
    chaves = lista[0].keys()
    return {c: float(np.std([m[c] for m in lista])) for c in chaves}


def _avaliar_folds(X, y, folds):
    metricas = []
    for tr, te in folds:
        modelo = clone(construir_pipeline(list(X.columns))).fit(X.iloc[tr], y.iloc[tr])
        metricas.append(resumo(y.iloc[te], modelo.predict_proba(X.iloc[te])[:, 1]))
    return {"media": _medio(metricas), "desvio": _desvio(metricas), "n_folds": len(metricas)}


def validar(df, features, ks_oot=(3, 6)):
    X = preparar_entrada(df, features)
    y = df[COL_ALVO]
    res = {}
    aleatoria = StratifiedKFold(5, shuffle=True, random_state=SEMENTE).split(X, y)
    res["aleatoria"] = _avaliar_folds(X, y, aleatoria)
    por_grupo = GroupKFold(5).split(X, y, groups=df[COL_ID])
    res["grupo_cliente"] = _avaliar_folds(X, y, por_grupo)
    safras = df[COL_SAFRA].tolist()
    for k in ks_oot:
        tr, te = split_out_of_time(safras, k)
        modelo = construir_pipeline(features).fit(X.iloc[tr], y.iloc[tr])
        m = resumo(y.iloc[te], modelo.predict_proba(X.iloc[te])[:, 1])
        m.update(n_treino=int(len(tr)), n_teste=int(len(te)), taxa_alvo_teste=float(y.iloc[te].mean()))
        res[f"out_of_time_k{k}"] = m
    return res


def executar():
    df = montar_desenvolvimento(carregar_bases())
    saida = {"taxa_alvo": float(df[COL_ALVO].mean()), "n": int(len(df)), "conjuntos": {}}
    for nome, feats in CONJUNTOS.items():
        print("Conjunto:", nome, feats)
        saida["conjuntos"][nome] = {"features": feats, **validar(df, feats)}
        for esquema, m in saida["conjuntos"][nome].items():
            if esquema == "features":
                continue
            v = m.get("media", m)
            print(f"  {esquema:20s} AUC {v['auc']:.4f}  KS {v['ks']:.4f}  Gini {v['gini']:.4f}")
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "validacao.json").write_text(
        json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return saida


if __name__ == "__main__":
    executar()
