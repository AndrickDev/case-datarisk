"""Monitoramento de drift: PSI por variavel e do escore por safra, KS entre safras.

Referencia das variaveis: safras de treino e calibracao do protocolo out-of-time.
Referencia do escore: safras de calibracao (predicao fora da amostra de treino).
Monitoradas: as safras do teste out-of-time e as safras da base de teste do desafio.
Uso: python -m src.drift
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.config import COL_ALVO, COL_SAFRA, FEATURES, FEATURES_CATEGORICAS, REPORTS_DIR
from src.data import carregar_bases, montar_desenvolvimento, montar_teste
from src.decisao import K_CALIBRACAO, K_TESTE
from src.features import construir_pipeline, preparar_entrada
from src.validacao import split_out_of_time

LIMITE_PSI_ALERTA = 0.25
LIMITE_PSI_ATENCAO = 0.10
EPS = 1e-4
# Variaveis de calendario variam por construcao entre safras; o PSI delas e estrutural.
VARIAVEIS_CALENDARIO = {"MES_EMISSAO", "DIA_EMISSAO", "DIA_VENCIMENTO", "TEMPO_CADASTRO_DIAS"}


def _psi_proporcoes(p_ref, p_atual):
    p_ref = np.clip(np.asarray(p_ref, dtype=float), EPS, None)
    p_atual = np.clip(np.asarray(p_atual, dtype=float), EPS, None)
    return float(np.sum((p_atual - p_ref) * np.log(p_atual / p_ref)))


def psi_numerico(ref, atual, n_bins=10):
    """PSI com faixas por quantis da referencia; nulos formam uma faixa propria."""
    ref, atual = pd.Series(ref), pd.Series(atual)
    cortes = np.unique(np.nanquantile(ref.dropna(), np.linspace(0, 1, n_bins + 1))[1:-1])

    def proporcoes(s):
        nulo = s.isna().mean()
        faixas = np.searchsorted(cortes, s.dropna().to_numpy(), side="right")
        cont = np.bincount(faixas, minlength=len(cortes) + 1) / max(len(s), 1)
        return np.append(cont, nulo)

    return _psi_proporcoes(proporcoes(ref), proporcoes(atual))


def psi_categorico(ref, atual):
    categorias = sorted(set(pd.Series(ref).astype(str)) | set(pd.Series(atual).astype(str)))
    pr = pd.Series(ref).astype(str).value_counts(normalize=True).reindex(categorias).fillna(0)
    pa = pd.Series(atual).astype(str).value_counts(normalize=True).reindex(categorias).fillna(0)
    return _psi_proporcoes(pr.to_numpy(), pa.to_numpy())


def psi_variavel(nome, ref, atual):
    if nome in FEATURES_CATEGORICAS:
        return psi_categorico(ref, atual)
    return psi_numerico(ref, atual)


def classificar(psi):
    if psi > LIMITE_PSI_ALERTA:
        return "alerta"
    if psi > LIMITE_PSI_ATENCAO:
        return "atencao"
    return "estavel"


def executar():
    bases = carregar_bases()
    dev = montar_desenvolvimento(bases)
    teste_desafio = montar_teste(bases)
    X = preparar_entrada(dev, FEATURES)
    y = dev[COL_ALVO]
    safras = dev[COL_SAFRA].tolist()

    ate_calib, te = split_out_of_time(safras, K_TESTE)
    tr_rel, cal_rel = split_out_of_time([safras[i] for i in ate_calib], K_CALIBRACAO)
    tr, cal = ate_calib[tr_rel], ate_calib[cal_rel]

    modelo = construir_pipeline(FEATURES).fit(X.iloc[tr], y.iloc[tr])
    ref_vars = X.iloc[ate_calib]
    ref_score = modelo.predict_proba(X.iloc[cal])[:, 1]

    monitor = pd.concat([
        X.iloc[te].assign(**{COL_SAFRA: dev[COL_SAFRA].iloc[te].to_numpy(), "origem": "desenvolvimento"}),
        preparar_entrada(teste_desafio, FEATURES).assign(
            **{COL_SAFRA: teste_desafio[COL_SAFRA].to_numpy(), "origem": "teste_desafio"}),
    ], ignore_index=True)
    monitor["ESCORE"] = modelo.predict_proba(monitor[FEATURES])[:, 1]

    psi_vars, psi_escore, ks_ref, ks_consec, alertas = {}, {}, {}, {}, []
    ordem = sorted(monitor[COL_SAFRA].unique())
    anterior = None
    for safra in ordem:
        grupo = monitor[monitor[COL_SAFRA] == safra]
        psi_vars[safra] = {f: psi_variavel(f, ref_vars[f], grupo[f]) for f in FEATURES}
        psi_escore[safra] = psi_numerico(ref_score, grupo["ESCORE"])
        ks_ref[safra] = float(ks_2samp(ref_score, grupo["ESCORE"]).statistic)
        if anterior is not None:
            ant = monitor.loc[monitor[COL_SAFRA] == anterior, "ESCORE"]
            ks_consec[f"{anterior}->{safra}"] = float(ks_2samp(ant, grupo["ESCORE"]).statistic)
        for f, v in psi_vars[safra].items():
            if v > LIMITE_PSI_ALERTA:
                tipo = "estrutural" if f in VARIAVEIS_CALENDARIO else "drift"
                alertas.append({"safra": safra, "variavel": f, "psi": v, "tipo": tipo})
        if psi_escore[safra] > LIMITE_PSI_ALERTA:
            alertas.append({"safra": safra, "variavel": "ESCORE", "psi": psi_escore[safra], "tipo": "drift"})
        anterior = safra

    saida = {
        "limites": {"alerta": LIMITE_PSI_ALERTA, "atencao": LIMITE_PSI_ATENCAO},
        "referencia_variaveis": sorted(set(dev[COL_SAFRA].iloc[ate_calib])),
        "referencia_escore": sorted(set(dev[COL_SAFRA].iloc[cal])),
        "safras_monitoradas": ordem,
        "origem_safras": monitor.groupby(COL_SAFRA)["origem"].first().to_dict(),
        "psi_variaveis": psi_vars,
        "psi_escore": psi_escore,
        "ks_escore_vs_referencia": ks_ref,
        "ks_escore_safras_consecutivas": ks_consec,
        "variaveis_calendario": sorted(VARIAVEIS_CALENDARIO),
        "alertas": alertas,
        "psi_maximo_por_variavel": {
            f: max(psi_vars[s][f] for s in ordem) for f in FEATURES
        },
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "drift.json").write_text(json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Safras monitoradas:", ordem)
    print("PSI do escore:", {s: round(v, 3) for s, v in psi_escore.items()})
    print("KS do escore vs referencia:", {s: round(v, 3) for s, v in ks_ref.items()})
    print("PSI maximo por variavel:", {f: round(v, 3) for f, v in saida["psi_maximo_por_variavel"].items()})
    reais = [a for a in alertas if a["tipo"] == "drift"]
    print("Alertas (PSI > 0,25):", len(alertas), "| drift:", len(reais), "| estruturais:", len(alertas) - len(reais))
    for a in reais:
        print("  ", a["safra"], a["variavel"], round(a["psi"], 3))
    return saida


if __name__ == "__main__":
    executar()
