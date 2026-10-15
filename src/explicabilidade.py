"""Explicabilidade: SHAP no LightGBM e IV / odds ratio na regressao logistica.

Convencao de WOE: ln(distribuicao de inadimplentes / distribuicao de adimplentes).
WOE maior indica faixa de maior risco; coeficientes positivos sao esperados.
Uso: python -m src.explicabilidade
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression

from src.config import COL_ALVO, COL_SAFRA, FEATURES, FEATURES_CATEGORICAS, REPORTS_DIR, SEMENTE
from src.data import carregar_bases, montar_desenvolvimento
from src.decisao import K_TESTE
from src.features import construir_pipeline, preparar_entrada
from src.metrics import resumo
from src.validacao import split_out_of_time

SUAVIZACAO = 0.5
N_AMOSTRA_SHAP = 5000


def faixas_variavel(serie, n_bins=10, categorica=False):
    """Retorna cortes (numerica) ou lista de categorias (categorica) ajustados na serie."""
    if categorica:
        return sorted(serie.astype(str).unique())
    return np.unique(np.nanquantile(serie.dropna(), np.linspace(0, 1, n_bins + 1))[1:-1])


def atribuir_faixa(serie, faixas, categorica=False):
    if categorica:
        return serie.astype(str).where(serie.astype(str).isin(faixas), "OUTROS")
    rotulos = pd.Series(np.searchsorted(faixas, serie.to_numpy(dtype=float), side="right"), index=serie.index)
    return rotulos.astype(str).where(serie.notna(), "NULO")


def tabela_woe(faixa, y):
    """WOE e IV por faixa, com suavizacao aditiva para faixas sem eventos."""
    t = pd.crosstab(faixa, y).reindex(columns=[0, 1], fill_value=0)
    ruim = t[1] + SUAVIZACAO
    bom = t[0] + SUAVIZACAO
    dist_ruim, dist_bom = ruim / ruim.sum(), bom / bom.sum()
    woe = np.log(dist_ruim / dist_bom)
    iv = float(((dist_ruim - dist_bom) * woe).sum())
    return woe.to_dict(), iv


class TransformadorWOE:
    def __init__(self, features):
        self.features = list(features)

    def fit(self, X, y):
        self.faixas_, self.woe_, self.iv_ = {}, {}, {}
        for f in self.features:
            cat = f in FEATURES_CATEGORICAS
            self.faixas_[f] = faixas_variavel(X[f], categorica=cat)
            woe, iv = tabela_woe(atribuir_faixa(X[f], self.faixas_[f], cat), y)
            self.woe_[f], self.iv_[f] = woe, iv
        return self

    def transform(self, X):
        saida = pd.DataFrame(index=X.index)
        for f in self.features:
            cat = f in FEATURES_CATEGORICAS
            faixa = atribuir_faixa(X[f], self.faixas_[f], cat)
            saida[f] = faixa.map(self.woe_[f]).fillna(0.0)
        return saida


def classificar_iv(iv):
    if iv < 0.02:
        return "nao preditiva"
    if iv < 0.10:
        return "fraca"
    if iv < 0.30:
        return "media"
    return "forte"


def executar():
    dev = montar_desenvolvimento(carregar_bases())
    X = preparar_entrada(dev, FEATURES)
    y = dev[COL_ALVO]
    ate, te = split_out_of_time(dev[COL_SAFRA].tolist(), K_TESTE)

    # SHAP no LightGBM treinado ate o inicio do teste out-of-time.
    pipe = construir_pipeline(FEATURES).fit(X.iloc[ate], y.iloc[ate])
    amostra = X.iloc[te].sample(min(N_AMOSTRA_SHAP, len(te)), random_state=SEMENTE)
    transformada = pipe.named_steps["prep"].transform(amostra)
    valores = shap.TreeExplainer(pipe.named_steps["modelo"]).shap_values(transformada)
    if isinstance(valores, list):
        valores = valores[1]
    importancia = pd.Series(np.abs(valores).mean(axis=0), index=transformada.columns).sort_values(ascending=False)
    plt.figure()
    shap.summary_plot(valores, transformada, show=False, max_display=14)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_resumo.png", dpi=110)
    plt.close()

    # IV e odds ratio na regressao logistica sobre variaveis em WOE (somente periodo de treino).
    woe = TransformadorWOE(FEATURES).fit(X.iloc[ate], y.iloc[ate])
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(woe.transform(X.iloc[ate]), y.iloc[ate])
    p_te = lr.predict_proba(woe.transform(X.iloc[te]))[:, 1]
    odds = pd.Series(np.exp(lr.coef_[0]), index=FEATURES)

    saida = {
        "shap_media_absoluta": importancia.to_dict(),
        "amostra_shap": int(len(amostra)),
        "iv": {f: {"iv": woe.iv_[f], "classe": classificar_iv(woe.iv_[f])} for f in FEATURES},
        "odds_ratio_por_unidade_de_woe": odds.to_dict(),
        "coeficientes": dict(zip(FEATURES, map(float, lr.coef_[0]))),
        "logistica_out_of_time": resumo(y.iloc[te], p_te),
    }
    (REPORTS_DIR / "explicabilidade.json").write_text(
        json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("SHAP |medio|:", {k: round(v, 4) for k, v in importancia.head(8).items()})
    print("IV:", {f: (round(v["iv"], 3), v["classe"]) for f, v in sorted(saida["iv"].items(), key=lambda kv: -kv[1]["iv"])})
    print("Odds ratio:", {k: round(v, 3) for k, v in odds.sort_values(ascending=False).items()})
    print("Logistica WOE out-of-time:", {k: round(v, 4) for k, v in saida["logistica_out_of_time"].items()})
    return saida


if __name__ == "__main__":
    executar()
