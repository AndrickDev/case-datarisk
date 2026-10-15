"""Metricas de classificacao, limiar por custo e calibracao de probabilidades.

Protocolo: o modelo e treinado nas safras mais antigas; as 3 safras seguintes
(anteriores ao teste) servem para calibrar e escolher limiar; o teste e out-of-time.
Uso: python -m src.decisao
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from src.config import COL_ALVO, COL_SAFRA, FEATURES, REPORTS_DIR
from src.data import carregar_bases, montar_desenvolvimento
from src.features import construir_pipeline, preparar_entrada
from src.validacao import split_out_of_time

K_TESTE = 6
K_CALIBRACAO = 3
RAZOES_CUSTO = (1, 2, 5, 10)  # custo de FN / custo de FP (hipoteses, nao dados do desafio)


def matriz_confusao(y, prob, limiar):
    y = np.asarray(y)
    pred = np.asarray(prob) >= limiar
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def precisao_recall(m):
    precisao = m["tp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) else 0.0
    recall = m["tp"] / (m["tp"] + m["fn"]) if (m["tp"] + m["fn"]) else 0.0
    return float(precisao), float(recall)


def custo_total(m, custo_fn, custo_fp):
    return m["fn"] * custo_fn + m["fp"] * custo_fp


def limiar_por_custo(y, prob, custo_fn, custo_fp, grade=None):
    """Limiar que minimiza custo_fn*FN + custo_fp*FP sobre uma grade de candidatos."""
    grade = np.unique(np.quantile(prob, np.linspace(0.01, 0.99, 197))) if grade is None else grade
    custos = [custo_total(matriz_confusao(y, prob, t), custo_fn, custo_fp) for t in grade]
    return float(grade[int(np.argmin(custos))])


def ajustar_platt(score_cal, y_cal):
    eps = 1e-6
    logit = lambda p: np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))  # noqa: E731
    lr = LogisticRegression(C=1e6, solver="lbfgs").fit(logit(score_cal).reshape(-1, 1), y_cal)
    return lambda p: lr.predict_proba(logit(np.asarray(p)).reshape(-1, 1))[:, 1]


def ajustar_isotonica(score_cal, y_cal):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(score_cal, y_cal)
    return lambda p: iso.predict(np.asarray(p))


def tabela_confiabilidade(y, prob, n_bins=10):
    freq, media = calibration_curve(y, prob, n_bins=n_bins, strategy="quantile")
    return {"prob_media": media.tolist(), "freq_observada": freq.tolist()}


def executar():
    df = montar_desenvolvimento(carregar_bases())
    X = preparar_entrada(df, FEATURES)
    y = df[COL_ALVO]
    safras = df[COL_SAFRA].tolist()
    ate_calib, te = split_out_of_time(safras, K_TESTE)
    sub_safras = [safras[i] for i in ate_calib]
    tr_rel, cal_rel = split_out_of_time(sub_safras, K_CALIBRACAO)
    tr, cal = ate_calib[tr_rel], ate_calib[cal_rel]

    modelo = construir_pipeline(FEATURES).fit(X.iloc[tr], y.iloc[tr])
    p_cal = modelo.predict_proba(X.iloc[cal])[:, 1]
    p_te = modelo.predict_proba(X.iloc[te])[:, 1]
    y_cal, y_te = y.iloc[cal].to_numpy(), y.iloc[te].to_numpy()

    platt = ajustar_platt(p_cal, y_cal)
    iso = ajustar_isotonica(p_cal, y_cal)
    versoes = {"bruto": p_te, "platt": platt(p_te), "isotonica": iso(p_te)}
    cal_versoes = {"bruto": p_cal, "platt": platt(p_cal), "isotonica": iso(p_cal)}

    calibracao = {
        nome: {
            "brier": float(brier_score_loss(y_te, p)),
            "media_prevista": float(p.mean()),
            "taxa_observada": float(y_te.mean()),
            "confiabilidade": tabela_confiabilidade(y_te, p),
        }
        for nome, p in versoes.items()
    }

    decisao = {"limiar_0_5": {}, "por_custo": {}}
    for nome, p in versoes.items():
        m = matriz_confusao(y_te, p, 0.5)
        pr, rc = precisao_recall(m)
        decisao["limiar_0_5"][nome] = {"limiar": 0.5, **m, "precisao": pr, "recall": rc}
    # Limiar por custo escolhido na janela de calibracao, avaliado no teste (probabilidade isotonica).
    for razao in RAZOES_CUSTO:
        t = limiar_por_custo(y_cal, cal_versoes["isotonica"], custo_fn=razao, custo_fp=1)
        m = matriz_confusao(y_te, versoes["isotonica"], t)
        pr, rc = precisao_recall(m)
        decisao["por_custo"][f"fn_{razao}x_fp"] = {
            "limiar": t, **m, "precisao": pr, "recall": rc,
            "custo_por_operacao": custo_total(m, razao, 1) / len(y_te),
        }

    saida = {
        "protocolo": {
            "n_treino": int(len(tr)), "n_calibracao": int(len(cal)), "n_teste": int(len(te)),
            "safras_teste": sorted(set(safras[i] for i in te)),
            "safras_calibracao": sorted(set(safras[i] for i in cal)),
            "taxa_alvo_teste": float(y_te.mean()),
        },
        "calibracao": calibracao,
        "decisao": decisao,
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "decisao_calibracao.json").write_text(
        json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfeita")
    for nome, p in versoes.items():
        c = calibracao[nome]["confiabilidade"]
        ax.plot(c["prob_media"], c["freq_observada"], marker="o", label=f"{nome} (Brier {calibracao[nome]['brier']:.4f})")
    ax.set_xlabel("Probabilidade prevista")
    ax.set_ylabel("Frequencia observada")
    ax.set_title("Curva de calibracao, teste out-of-time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "calibracao.png", dpi=110)

    print(json.dumps({k: {n: v["brier"] for n, v in calibracao.items()} for k in ["brier"]}, indent=2))
    for nome, d in decisao["limiar_0_5"].items():
        print("limiar 0,5", nome, {k: round(v, 4) if isinstance(v, float) else v for k, v in d.items()})
    for nome, d in decisao["por_custo"].items():
        print("custo", nome, {k: round(v, 4) if isinstance(v, float) else v for k, v in d.items()})
    return saida


if __name__ == "__main__":
    executar()
